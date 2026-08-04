use eyre::Context;
use grahf::{
    components::{initialization, mutation, normalization, recombination},
    problems::algorithm_design::{AlgorithmDesignProblem, TunedParams},
};
use log::{debug, info};
use mahf::{
    components::{archive, replacement, selection},
    conditions,
    logging::Logger,
    problems::KnownOptimumProblem,
    Condition, Configuration, ExecResult, SingleObjectiveProblem, State,
};
use serde::Deserialize;

/// Logs the best individual's graph structure and objective value at the current iteration.
pub fn debug_best_individual<P: SingleObjectiveProblem + KnownOptimumProblem>(
    problem: &AlgorithmDesignProblem<P>,
    state: &mut State<AlgorithmDesignProblem<P>>,
) {
    let i = state.iterations();
    let best = state.best_individual().unwrap();
    let best_objective_value = best.objective().value();
    debug!("Best Individual:\n{}", problem.id_graph(best.solution()));
    debug!("Tuning: {:?}", best.state().borrow_value::<TunedParams>());
    info!("Finished iteration {i} with f(best) = {best_objective_value}.");
}

/// Parameters for [`graph_ga`].
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct GrahfParameters {
    /// The maximal number of nodes for initial graphs.
    pub max_initial_nodes: u32,
    /// The probability of an edge existing for initial graphs.
    pub initial_edge_p: f64,
    /// The population size.
    pub population_size: u32,
    /// The tournament size.
    pub tournament_size: u32,
    /// The archive size.
    pub archive_size: u32,
    /// The frequency of elitist insertion.
    pub elitist_freq: u32,
    /// The crossover probability.
    pub pc: f64,
    /// The node insertion/removal mutation rate.
    pub rm_node: f64,
    /// The edge insertion/removal mutation rate.
    pub rm_edge: f64,
    /// The node weight mutation rate.
    pub rm_node_weight: f64,
    /// The edge weight mutation rate.
    pub rm_edge_weight: f64,
}

/// A single-objective GA operating on a directed graph search space.
pub fn grahf<P>(
    params: GrahfParameters,
    condition: Box<dyn Condition<AlgorithmDesignProblem<P>>>,
) -> ExecResult<Configuration<AlgorithmDesignProblem<P>>>
where
    P: SingleObjectiveProblem + KnownOptimumProblem,
{
    let GrahfParameters {
        max_initial_nodes,
        initial_edge_p,
        population_size,
        tournament_size,
        archive_size,
        elitist_freq,
        pc,
        rm_node,
        rm_edge,
        rm_node_weight,
        rm_edge_weight,
    } = params;

    Ok(Configuration::builder()
        .debug(|_, _| info!("Initializing population."))
        .do_(
            initialization::UniformRandomGraph::new(
                max_initial_nodes,
                initial_edge_p,
                population_size,
            )
            .wrap_err("failed to construct initialization")?,
        )
        .do_(replacement::Merge::new())
        .evaluate()
        .update_best_individual()
        .do_(archive::ElitistArchiveUpdate::new(archive_size as usize))
        .debug(|_, _| info!("Finished initialization."))
        .debug(debug_best_individual)
        .while_(condition, |builder| {
            builder
                .debug(|_, state| info!("Starting iteration {}.", state.iterations()))
                .do_(selection::Tournament::new(population_size, tournament_size))
                .debug(|_, _| info!("Doing crossover."))
                .do_(recombination::GraphPartitionCrossover::new_insert_both(pc))
                .debug(|_, _| info!("Doing mutations."))
                .do_(mutation::NodeInsertion::new(rm_node))
                .do_(mutation::EdgeInsertion::new(rm_edge))
                .do_(mutation::NodeWeightMutation::new(rm_node_weight))
                .do_(mutation::EdgeWeightMutation::new(rm_edge_weight))
                .debug(|_, _| info!("Starting evaluation."))
                .evaluate()
                .do_(normalization::NormalizeBestIndividual::new())
                .update_best_individual()
                .do_(normalization::NormalizeArchive::new())
                .do_(archive::ElitistArchiveUpdate::new(archive_size as usize))
                .do_(normalization::NormalizeParents::new())
                .debug(|_, _| info!("Doing replacement."))
                .do_(replacement::Generational::new(population_size))
                .if_(conditions::EveryN::iterations(elitist_freq), |builder| {
                    builder
                        .debug(|_, _| info!("Inserting random elitist from archive."))
                        .do_(archive::RandomElitistIntoPopulation::new())
                })
                .do_(mahf::components::initialization::Empty::new())
                .do_(replacement::MuPlusLambda::new(population_size))
                .debug(debug_best_individual)
                .do_(Logger::new())
        })
        .build())
}
