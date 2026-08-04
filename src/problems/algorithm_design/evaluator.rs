//! Evaluator that turns an island graph into a runnable metaheuristic and scores it.

use std::sync::Arc;

use derive_more::{Deref, DerefMut};
use itertools::izip;
use log::{debug, error, info, log_enabled, trace, warn};
use mahf::{
    conditions,
    params::Params,
    problems::{Evaluate, KnownOptimumProblem},
    Individual, Problem, SingleObjective, SingleObjectiveProblem, State,
};
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    components::transform::SolutionTransformer,
    problems::{
        algorithm_design::{
            statistics::{
                standardized_objective_with_stats, GlobalEvaluationStatistics,
                MultiProblemEvaluation, Statistics,
            },
            tuning::par_tune,
            AlgorithmDesignProblem,
        },
        ConstrainedProblem,
    },
};

/// The per-instance performance of an evaluated island graph, stored on the individual.
#[derive(Clone, Debug, Deref, DerefMut)]
pub struct EvaluationStatistics(pub MultiProblemEvaluation);

/// The IRACE-tuned parameters of an evaluated island graph, stored on the individual.
#[derive(Clone, Deref, DerefMut)]
pub struct TunedParams(pub Params);

/// Evaluates island graph individuals by tuning and running them on the inner problem.
pub struct MetaheuristicIslandDesignEvaluator<O: Evaluate> {
    /// Evaluators for the inner problem, one per instance.
    pub evaluators: Vec<O>,
    /// Optional transformer for dimension-aware migration during evaluation.
    pub transformer: Option<Arc<dyn SolutionTransformer<O::Problem>>>,
}

impl<O: Evaluate> MetaheuristicIslandDesignEvaluator<O> {
    /// Creates a new evaluator without a dimension transformer.
    ///
    /// # Arguments
    ///
    /// * `evaluators` - One evaluator per problem instance.
    ///
    /// # Returns
    ///
    /// The evaluator.
    pub fn new(evaluators: impl IntoIterator<Item = O>) -> Self {
        Self {
            evaluators: evaluators.into_iter().collect(),
            transformer: None,
        }
    }

    /// Creates a new evaluator with a transformer for dimension-aware migration.
    ///
    /// # Arguments
    ///
    /// * `evaluators` - One evaluator per problem instance.
    /// * `transformer` - Resizes migrants between islands of different dimensions.
    ///
    /// # Returns
    ///
    /// The evaluator.
    pub fn with_transformer(
        evaluators: impl IntoIterator<Item = O>,
        transformer: Arc<dyn SolutionTransformer<O::Problem>>,
    ) -> Self {
        Self {
            evaluators: evaluators.into_iter().collect(),
            transformer: Some(transformer),
        }
    }
}

impl<O> Evaluate for MetaheuristicIslandDesignEvaluator<O>
where
    O: Evaluate + Clone + 'static,
    O::Problem: SingleObjectiveProblem + KnownOptimumProblem + Send + Sync,
    <O::Problem as Problem>::Encoding: AsRef<[f64]>,
{
    type Problem = AlgorithmDesignProblem<O::Problem>;

    fn evaluate(
        &mut self,
        problem: &Self::Problem,
        state: &mut State<Self::Problem>,
        individuals: &mut [Individual<Self::Problem>],
    ) {
        // R supports only up to `i32::MAX` integer values.
        let tuning_seed = state.random_mut().gen_range(0..(i32::MAX as u32));

        let (feasible_indices, feasible_individuals): (Vec<_>, Vec<_>) = individuals
            .iter_mut()
            .enumerate()
            .filter_map(|(i, individual)| {
                if problem.feasible(individual.solution()) {
                    Some((i, individual))
                } else {
                    error!("Individual {i} is not feasible, skipping evaluation.");
                    individual.set_objective(SingleObjective::INFINITY);
                    None
                }
            })
            .unzip();

        // Construct the param space and builder for each individual.
        let condition = conditions::LessThanN::evaluations(problem.max_evaluations)
            & !conditions::OptimumReached::new(problem.epsilon).unwrap();

        let (param_spaces, builders): (Vec<_>, Vec<_>) = feasible_individuals
            .iter()
            .map(|individual| problem.builder_graph(individual.solution()))
            .map(|builder_graph| {
                let param_space = builder_graph.param_space();
                (param_space, builder_graph.into_builder(condition.clone()))
            })
            .unzip();

        // Tune all individuals.
        info!("Starting parallel tuning of population with seed {tuning_seed}.");
        let tuned_params = par_tune(
            builders.clone(),
            param_spaces,
            problem.tuning_scenario.clone(),
            &problem.instances,
            self.evaluators.clone(),
            problem.performance_measure.clone(),
            problem.num_tuning_repetitions,
            // At least one job, even when the tuning scenario claims every available thread.
            (rayon::current_num_threads() / problem.tuning_scenario.num_jobs.max(1)).max(1),
            tuning_seed,
            self.transformer.clone(),
        )
        .expect("error while tuning");

        // Evaluate all individuals.
        info!("Starting parallel evaluation of population.");
        let rngs: Vec<_> = state
            .random_mut()
            .iter_children()
            .take(feasible_individuals.len())
            .collect();
        let evaluators: Vec<_> =
            std::iter::repeat_n(self.evaluators.clone(), builders.len()).collect();

        let transformer = self.transformer.clone();
        let multi_evaluations: Vec<_> = (tuned_params.clone(), builders, evaluators, rngs)
            .into_par_iter()
            .map(|(params, builder, evaluators, mut rng)| {
                let transformer_clone = transformer.clone();
                builder(params).and_then(|config| {
                    let evaluations = problem.evaluate_with_transformer(
                        &config,
                        evaluators,
                        &mut rng,
                        transformer_clone,
                    )?;
                    trace!("Finished all evaluations for individual.");
                    Ok(evaluations)
                })
            })
            .collect();

        // ORDERING: snapshot the statistics computed from *previous* generations before
        // inserting the current one. Standardizing an individual against a baseline that
        // already includes that individual biases the z-scores toward zero and makes
        // cross-generation comparisons inconsistent.
        info!("Updating global statistics.");
        let mut global_statistics = state
            .entry::<GlobalEvaluationStatistics>()
            .or_insert(GlobalEvaluationStatistics::new(problem.instances.len()));

        let pre_update_statistics: Vec<Option<Statistics>> = global_statistics.get();

        for multi_evaluation in multi_evaluations.iter().flatten() {
            global_statistics.push(multi_evaluation);
        }
        global_statistics.update();

        if log_enabled!(log::Level::Debug) {
            for (i, (instance, statistics)) in
                izip!(&problem.instances, global_statistics.get()).enumerate()
            {
                match statistics {
                    Some(Statistics {
                        median,
                        deviation,
                        len,
                    }) => debug!(
                        "{} ({i}): {} = {median} ± {deviation} ({len} values)",
                        instance.name(),
                        problem.performance_measure.name(),
                    ),
                    None => warn!("Empty evaluations for {}.", instance.name()),
                }
            }
        }

        // Record the tuning result and the standardized objective on each individual.
        for (individual_index, multi_evaluation, params, individual) in izip!(
            feasible_indices,
            multi_evaluations,
            tuned_params,
            feasible_individuals
        ) {
            debug!(
                "Individual {individual_index}:\n{}",
                problem.id_graph(individual.solution())
            );
            debug!("Tuning: {params:?}");

            individual.state_mut().insert(TunedParams(params.as_nest()));

            match multi_evaluation {
                Ok(multi_evaluation) => {
                    for (problem_index, (evaluation, instance)) in
                        izip!(&multi_evaluation, &problem.instances).enumerate()
                    {
                        if evaluation.is_none() {
                            warn!("Failed all runs on {} ({problem_index}).", instance.name());
                        }
                    }

                    // Standardize against the pre-update snapshot so the current generation
                    // is never part of its own baseline.
                    let objective_value = standardized_objective_with_stats(
                        &pre_update_statistics,
                        &multi_evaluation,
                        problem.penalty,
                    );

                    debug!("f(x_{individual_index}) = {}", objective_value.value());
                    individual.set_objective(objective_value);
                    individual
                        .state_mut()
                        .insert(EvaluationStatistics(multi_evaluation));
                }
                Err(e) => {
                    error!("Evaluation failed: {e:?}");
                    individual.set_objective(SingleObjective::INFINITY);
                }
            }
        }
    }
}
