//! Island graph executor and supporting state types for running heterogeneous island models.

use std::sync::Arc;

use better_any::{Tid, TidAble};
use derivative::Derivative;
use derive_more::{Deref, DerefMut};
use eyre::{ensure, ContextCompat};
use itertools::izip;
use mahf::{
    lens::ValueOf,
    logging::{Log, LogConfig},
    prelude::StateReq,
    state::common,
    Component, CustomState, ExecResult, Individual, Problem, SingleObjectiveProblem, State,
};
use petgraph::visit::EdgeRef;
use serde::Serialize;

use crate::{
    components::transform::{SolutionTransformer, TransformRequest},
    graph::di::DiGraph,
    problems::{algorithm_design::islands::Migration, evaluate::DummyEvaluator},
};

/// Holds the solution transformer in island state so migration can resize solutions.
#[derive(Tid)]
pub struct MigrationTransformer<P: Problem + 'static>(pub Arc<dyn SolutionTransformer<P>>);

impl<P: Problem> CustomState<'_> for MigrationTransformer<P> {}

/// Per-island logging configurations, stored in the parent state.
#[derive(Tid, Deref, DerefMut)]
pub struct IslandLogConfig<P: Problem + 'static>(pub Vec<LogConfig<P>>);

impl<P: Problem> CustomState<'_> for IslandLogConfig<P> {}

/// The collection of per-island states, owned by the parent state.
#[derive(Tid, Deref, DerefMut)]
pub struct IslandStates<'a, P: Problem + 'static>(Vec<State<'a, P>>);

impl<'a, P: Problem> CustomState<'a> for IslandStates<'a, P> {}

/// Initializes each island's state and population before the main loop starts.
#[derive(Serialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = ""))]
pub struct IslandStatesInit<P: Problem> {
    /// One initialization component per island, in graph node order.
    pub initializations: Vec<Box<dyn Component<P>>>,
}

impl<P: Problem> IslandStatesInit<P> {
    /// Creates a new `IslandStatesInit` from an iterator of island initialization components.
    pub fn from_params(initializations: impl IntoIterator<Item = Box<dyn Component<P>>>) -> Self {
        Self {
            initializations: initializations.into_iter().collect(),
        }
    }

    /// Wraps `IslandStatesInit` in a boxed component.
    pub fn new(
        initializations: impl IntoIterator<Item = Box<dyn Component<P>>>,
    ) -> Box<dyn Component<P>> {
        Box::new(Self::from_params(initializations))
    }
}

impl<P> Component<P> for IslandStatesInit<P>
where
    P: Problem,
{
    fn init(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        // Create a state for each island.
        let mut states: Vec<_> = state
            .random_mut()
            .iter_children()
            .take(self.initializations.len())
            .map(|rng| {
                let mut state = State::new();
                state.insert(rng);
                state.insert(common::Populations::<P>::new());
                state.insert(common::Iterations(0));
                state.insert_evaluator(DummyEvaluator::default());
                state.insert(Log::new());
                state
            })
            .collect();

        if let Ok(island_configs) = state.try_get_value::<IslandLogConfig<P>>() {
            for (log_config, state) in izip!(island_configs, &mut states) {
                state.insert(log_config);
            }
        }

        for (initialization, state) in izip!(&self.initializations, &mut states) {
            initialization.init(problem, state)?;
        }

        state.insert(IslandStates(states));

        Ok(())
    }

    fn require(&self, problem: &P, state_req: &StateReq<P>) -> ExecResult<()> {
        let states = state_req.state().borrow_value::<IslandStates<P>>();

        for (initialization, state) in izip!(&self.initializations, &*states) {
            let state_req = state.requirements();
            initialization.require(problem, &state_req)?;
        }

        Ok(())
    }

    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        let mut evaluator = state.remove::<common::Evaluator<P>>()?;
        let mut states = state.borrow_value_mut::<IslandStates<P>>();

        for (initialization, state) in izip!(&self.initializations, &mut *states) {
            state.insert(evaluator);
            initialization.execute(problem, state)?;
            evaluator = state.take::<common::Evaluator<P>>();
        }

        drop(states);
        state.insert(evaluator);

        Ok(())
    }
}

pub type IslandGraph<P> = DiGraph<Box<dyn Component<P>>, Migration<P>>;

/// Runs all islands for one iteration and migrates individuals between them.
#[derive(Serialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = ""))]
pub struct IslandGraphExecutor<P: Problem> {
    /// The island graph: nodes are island algorithms, edges are migration policies.
    pub islands: IslandGraph<P>,
}

impl<P: Problem> IslandGraphExecutor<P>
where
    P::Encoding: AsRef<[f64]>,
{
    /// Creates a new `IslandGraphExecutor` wrapping the given island graph.
    pub fn from_params(graph: IslandGraph<P>) -> Self {
        Self { islands: graph }
    }

    /// Wraps `IslandGraphExecutor` in a boxed component.
    pub fn new(graph: IslandGraph<P>) -> Box<dyn Component<P>> {
        Box::new(Self::from_params(graph))
    }
}

impl<P> Component<P> for IslandGraphExecutor<P>
where
    P: Problem,
    P::Encoding: AsRef<[f64]>,
{
    fn init(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        state.insert(common::Evaluations(0));

        let mut states = state.borrow_value_mut::<IslandStates<P>>();

        for (island, state) in izip!(self.islands.node_weights(), &mut *states) {
            state.insert(common::Progress::<ValueOf<common::Evaluations>>::default());
            island.init(problem, state)?;
        }

        for edge in self.islands.edge_references() {
            let migration = edge.weight();

            let source_state = &mut states[edge.source().index()];
            migration.condition.init(problem, source_state)?;
            migration.selection.init(problem, source_state)?;

            let target_state = &mut states[edge.target().index()];
            migration.replacement.init(problem, target_state)?;
        }

        Ok(())
    }

    fn require(&self, problem: &P, state_req: &StateReq<P>) -> ExecResult<()> {
        let states = state_req.state().borrow_value::<IslandStates<P>>();

        for (island, state) in izip!(self.islands.node_weights(), &*states) {
            island.require(problem, &state.requirements())?;
        }

        for edge in self.islands.edge_references() {
            let migration = edge.weight();

            let source_state = &states[edge.source().index()];
            let source_state_req = source_state.requirements();
            migration.condition.require(problem, &source_state_req)?;
            migration.selection.require(problem, &source_state_req)?;

            let target_state = &states[edge.target().index()];
            let target_state_req = target_state.requirements();
            migration.replacement.require(problem, &target_state_req)?;
        }

        Ok(())
    }

    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        let mut evaluator = state.remove::<common::Evaluator<P>>()?;
        let mut states = state.borrow_value_mut::<IslandStates<P>>();

        let progress = state.get_value::<common::Progress<ValueOf<common::Evaluations>>>();

        for (island, island_state) in izip!(self.islands.node_weights(), &mut *states) {
            island_state.insert(evaluator);
            island_state.set_value::<common::Progress<ValueOf<common::Evaluations>>>(progress);

            island.execute(problem, island_state)?;

            evaluator = island_state.take::<common::Evaluator<P>>();
        }

        // Transfer evaluations into parent state.
        let island_evaluations = states
            .iter()
            .filter_map(|state| state.try_get_value::<common::Evaluations>().ok())
            .sum();
        let old_island_evaluations = state.get_value::<common::Evaluations>();
        ensure!(
            island_evaluations > old_island_evaluations,
            "no evaluations in iteration"
        );
        state.set_value::<common::Evaluations>(island_evaluations);

        // Migrate individuals between islands.
        let maybe_transformer = state.try_borrow::<MigrationTransformer<P>>().ok();

        for edge in self.islands.edge_references() {
            let migration = edge.weight();
            let source_idx = edge.source().index();
            let target_idx = edge.target().index();

            let source_state = &mut states[source_idx];
            if migration.condition.evaluate(problem, source_state)? {
                migration.selection.execute(problem, source_state)?;
                let mut selected = source_state.populations_mut().pop();

                // Transform migrants if dimensions differ.
                //
                // Whether to transform at all is decided from the *first* migrant's actual
                // solution length rather than from the IRACE-tuned `source_dimension`,
                // because a migrant can arrive at a different length than the stored
                // parameter suggests (e.g. after a prior cascaded migration). Each migrant
                // is then transformed from its own length, so a population that mixes
                // lengths is still resized correctly once the gate opens.
                let did_transform = match (
                    &migration.transform_method,
                    migration.target_dimension,
                    &maybe_transformer,
                ) {
                    (Some(method), Some(target_dim), Some(transformer)) => {
                        let actual_source_dim = selected
                            .first()
                            .map(|individual| individual.solution().as_ref().len() as u32)
                            .unwrap_or(target_dim);

                        if actual_source_dim != target_dim && !selected.is_empty() {
                            let mut rng = source_state.random_mut();

                            for individual in selected.iter_mut() {
                                let source_dim = individual.solution().as_ref().len() as u32;
                                let solution = transformer.0.transform(
                                    problem,
                                    individual.solution(),
                                    TransformRequest::new(source_dim, target_dim, method),
                                    &mut rng,
                                );
                                *individual = Individual::new_unevaluated(solution);
                            }
                            true
                        } else {
                            false
                        }
                    }
                    _ => false,
                };

                let target_state = &mut states[target_idx];

                // Only re-evaluate if a transformation occurred: transformed individuals
                // are unevaluated and replacement strategies (e.g. `MuPlusLambda`) require
                // objective values.
                if did_transform {
                    target_state.insert(evaluator);
                    let mut target_evaluator = target_state.take::<common::Evaluator<P>>();
                    target_evaluator
                        .as_inner_mut()
                        .evaluate(problem, target_state, &mut selected);
                    target_state.insert(target_evaluator);
                    evaluator = target_state.take::<common::Evaluator<P>>();
                }

                target_state.populations_mut().push(selected);

                migration.replacement.execute(problem, target_state)?;
            }
        }

        drop(states);
        drop(maybe_transformer);
        state.insert(evaluator);

        Ok(())
    }
}

#[derive(Serialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = ""))]
pub struct UpdateBestIslandIndividual;

impl UpdateBestIslandIndividual {
    /// Creates a new `UpdateBestIslandIndividual`.
    pub fn from_params() -> Self {
        Self
    }

    /// Wraps `UpdateBestIslandIndividual` in a boxed component.
    pub fn new<P: SingleObjectiveProblem>() -> Box<dyn Component<P>> {
        Box::new(Self::from_params())
    }
}

impl<P: SingleObjectiveProblem> Component<P> for UpdateBestIslandIndividual {
    fn init(&self, _problem: &P, state: &mut State<P>) -> ExecResult<()> {
        state.insert(common::BestIndividual::<P>::new());
        Ok(())
    }

    fn require(&self, _problem: &P, state_req: &StateReq<P>) -> ExecResult<()> {
        state_req.require::<Self, IslandStates<P>>()?;
        Ok(())
    }

    fn execute(&self, _problem: &P, state: &mut State<P>) -> ExecResult<()> {
        let states = state.borrow_value_mut::<IslandStates<P>>();
        let best = states
            .iter()
            .filter_map(|state| state.best_individual())
            .min_by_key(|i| *i.objective())
            .wrap_err("no island has a best individual")?;
        state
            .borrow_mut::<common::BestIndividual<P>>()
            .update(&*best);

        Ok(())
    }
}
