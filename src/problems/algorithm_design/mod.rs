//! Graph-based algorithm design.
//!
//! An algorithm is encoded as a directed island graph: node weights select the island
//! metaheuristic, edge weights select the migration policy. The encoded algorithm is tuned
//! with IRACE and then run on every problem instance to obtain its fitness.

use std::{collections::HashSet, sync::Arc};

use irace_rs::scenario::Scenario;
use log::{log_enabled, trace};
use mahf::{
    problems::{Evaluate, KnownOptimumProblem},
    Configuration, ExecResult, Problem, Random, SingleObjective, SingleObjectiveProblem,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use typed_builder::TypedBuilder;

use crate::{
    components::{island::MigrationTransformer, transform::SolutionTransformer},
    graph::DiGraph,
    problems::{
        algorithm_design::{
            builder::{BuilderGraph, MetaheuristicIslandBuilder, MigrationBuilder},
            performance::PerformanceMeasure,
        },
        ConstrainedProblem, DiIntGraph, DirectedGraphProblem,
    },
};

pub mod builder;
pub mod evaluator;
pub mod islands;
pub mod performance;
pub mod statistics;
pub mod tuning;

pub use evaluator::{EvaluationStatistics, MetaheuristicIslandDesignEvaluator, TunedParams};
pub use statistics::{
    standardized_objective_with_stats, GlobalEvaluationStatistics, MultiProblemEvaluation,
    Statistics, StatisticsTracker,
};

/// Island model algorithm design.
///
/// Algorithms are encoded as directed island graphs with integer weights on nodes and edges,
/// which decide which island (node) and migration (edge) types are used. The generated
/// algorithm is then evaluated on the problem `P`.
#[derive(TypedBuilder)]
pub struct AlgorithmDesignProblem<P: SingleObjectiveProblem + KnownOptimumProblem> {
    /// The maximal difference between best objective value and optimum.
    #[builder(default = 1e-6)]
    pub epsilon: f64,
    /// The maximal amount of evaluations until termination.
    #[builder(default = 200_000)]
    pub max_evaluations: u32,
    /// The algorithm performance measure.
    pub performance_measure: Box<dyn PerformanceMeasure<P>>,
    /// The penalty value for failing all runs on a problem in standardized space.
    #[builder(default = 100.0)]
    pub penalty: f64,
    /// The number of repetitions for evaluating on a single instance.
    #[builder(default = 30)]
    pub num_repetitions: u32,
    /// The number of repetitions for evaluating a configuration on a single instance.
    #[builder(default = 10)]
    pub num_tuning_repetitions: u32,
    /// The enumerated node type.
    pub island_builders: Vec<Box<dyn MetaheuristicIslandBuilder<P>>>,
    /// Ids of node types that can not exist in isolation, e.g. archives.
    #[builder(setter(into))]
    pub dependent_island_builders: HashSet<String>,
    /// The enumerated edge type.
    pub migration_builders: Vec<Box<dyn MigrationBuilder<P>>>,
    /// The instances the algorithm is evaluated on.
    pub instances: Vec<Arc<P>>,
    /// The scenario used for tuning.
    #[builder(setter(into))]
    pub tuning_scenario: Arc<Scenario>,
}

impl<P> AlgorithmDesignProblem<P>
where
    P: SingleObjectiveProblem + KnownOptimumProblem,
{
    /// Returns whether all node weights map to valid, non-exclusively-dependent island types.
    ///
    /// # Arguments
    ///
    /// * `solution` - The encoded island graph.
    ///
    /// # Returns
    ///
    /// `true` if every node weight is a known island type and at least one node is
    /// independent (a graph consisting only of archives cannot optimize anything).
    pub fn nodes_valid(&self, solution: &DiIntGraph) -> bool {
        let weights: HashSet<_> = solution.node_weights().copied().collect();
        let types: HashSet<_> = self.node_types().into_iter().collect();

        if !weights.is_subset(&types) {
            return false;
        }

        // Indexing is safe because every weight was just checked against the node types.
        !weights
            .iter()
            .map(|&weight| self.island_builders[weight as usize].id())
            .all(|id| self.dependent_island_builders.contains(&id))
    }

    /// Returns whether all edge weights map to valid migration types.
    ///
    /// # Arguments
    ///
    /// * `solution` - The encoded island graph.
    ///
    /// # Returns
    ///
    /// `true` if every edge weight is a known migration type.
    pub fn edges_valid(&self, solution: &DiIntGraph) -> bool {
        let weights: HashSet<_> = solution.edge_weights().copied().collect();
        let types: HashSet<_> = self.edge_types().into_iter().collect();

        weights.is_subset(&types)
    }

    /// Maps each node and edge weight in `solution` to its corresponding builder.
    ///
    /// # Arguments
    ///
    /// * `solution` - The encoded island graph; must be feasible.
    ///
    /// # Returns
    ///
    /// The graph of island and migration builders.
    ///
    /// # Panics
    ///
    /// Panics if a weight is out of range; call [`ConstrainedProblem::feasible`] first.
    pub fn builder_graph(&self, solution: &DiIntGraph) -> BuilderGraph<P> {
        solution.map(
            |_, &weight| self.island_builders[weight as usize].clone(),
            |_, &weight| self.migration_builders[weight as usize].clone(),
        )
    }

    /// Maps each node and edge weight in `solution` to its builder's string id, for logging.
    ///
    /// # Arguments
    ///
    /// * `solution` - The encoded island graph; must be feasible.
    ///
    /// # Returns
    ///
    /// The graph of builder ids.
    ///
    /// # Panics
    ///
    /// Panics if a weight is out of range; call [`ConstrainedProblem::feasible`] first.
    pub fn id_graph(&self, solution: &DiIntGraph) -> DiGraph<String, String> {
        solution.map(
            |i, &weight| {
                format!(
                    "{}:{}",
                    i.index(),
                    self.island_builders[weight as usize].id()
                )
            },
            |_, &weight| self.migration_builders[weight as usize].id(),
        )
    }
}

impl<P> AlgorithmDesignProblem<P>
where
    P: SingleObjectiveProblem + KnownOptimumProblem + Send + Sync,
{
    /// Evaluates `config` on all instances without dimension transformation.
    ///
    /// # Arguments
    ///
    /// * `config` - The metaheuristic to run.
    /// * `evaluators` - One evaluator per instance.
    /// * `rng` - Random number generator seeding the repetitions.
    ///
    /// # Returns
    ///
    /// One performance value per instance; `None` where all repetitions failed.
    ///
    /// # Errors
    ///
    /// Returns an error if any repetition fails to execute.
    pub fn evaluate<O>(
        &self,
        config: &Configuration<P>,
        evaluators: Vec<O>,
        rng: &mut Random,
    ) -> ExecResult<Vec<Option<f64>>>
    where
        O: Evaluate<Problem = P> + Clone + 'static,
    {
        self.evaluate_with_transformer(config, evaluators, rng, None)
    }

    /// Evaluates `config` on all instances, using `transformer` for dimension-aware migration.
    ///
    /// # Arguments
    ///
    /// * `config` - The metaheuristic to run.
    /// * `evaluators` - One evaluator per instance.
    /// * `rng` - Random number generator seeding the repetitions.
    /// * `transformer` - Resizes migrants between islands of different dimensions.
    ///
    /// # Returns
    ///
    /// One performance value per instance; `None` where all repetitions failed.
    ///
    /// # Errors
    ///
    /// Returns an error if any repetition fails to execute.
    pub fn evaluate_with_transformer<O>(
        &self,
        config: &Configuration<P>,
        evaluators: Vec<O>,
        rng: &mut Random,
        transformer: Option<Arc<dyn SolutionTransformer<P>>>,
    ) -> ExecResult<Vec<Option<f64>>>
    where
        O: Evaluate<Problem = P> + Clone + 'static,
    {
        let rngs: Vec<_> = rng.iter_children().take(self.instances.len()).collect();

        (&self.instances, evaluators, rngs)
            .into_par_iter()
            .map(|(problem, evaluator, mut rng)| {
                let transformer = transformer.clone();

                let states: Vec<_> = rng
                    .iter_children()
                    .take(self.num_repetitions as usize)
                    .map(|rng| {
                        config.optimize_with(problem, |state| {
                            state.insert_evaluator(evaluator.clone());
                            state.insert(rng);
                            if let Some(transformer) = transformer.clone() {
                                state.insert(MigrationTransformer(transformer));
                            }
                            Ok(())
                        })
                    })
                    .collect::<Result<_, _>>()?;

                let performance = self.performance_measure.measure(&states);

                if log_enabled!(log::Level::Trace) {
                    match performance {
                        Some(performance) => trace!(
                            "Finished evaluation of {} successfully with an {} of {}",
                            problem.name(),
                            self.performance_measure.name(),
                            performance
                        ),
                        None => trace!(
                            "Finished evaluation of {} with no successful runs.",
                            problem.name()
                        ),
                    }
                }

                Ok(performance)
            })
            .collect()
    }
}

impl<P> Problem for AlgorithmDesignProblem<P>
where
    P: SingleObjectiveProblem + KnownOptimumProblem,
{
    type Encoding = DiIntGraph;
    type Objective = SingleObjective;

    fn name(&self) -> &str {
        "Algorithm Graph Design Problem"
    }
}

impl<P> DirectedGraphProblem for AlgorithmDesignProblem<P>
where
    P: SingleObjectiveProblem + KnownOptimumProblem,
{
    fn node_types(&self) -> Vec<u32> {
        (0..self.island_builders.len() as u32).collect()
    }

    fn edge_types(&self) -> Vec<u32> {
        (0..self.migration_builders.len() as u32).collect()
    }
}

impl<P> ConstrainedProblem for AlgorithmDesignProblem<P>
where
    P: SingleObjectiveProblem + KnownOptimumProblem,
{
    fn feasible(&self, solution: &Self::Encoding) -> bool {
        let empty_graph = solution.node_count() == 0;
        let nodes_valid = self.nodes_valid(solution);
        let edges_valid = self.edges_valid(solution);

        !empty_graph && nodes_valid && edges_valid
    }
}
