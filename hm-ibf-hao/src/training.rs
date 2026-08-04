//! GRAHF structure search: finds an island graph for the horizontal alignment benchmark.

use std::{collections::HashMap, fs, io::BufWriter, path::Path, sync::Arc};

use eyre::WrapErr;
use grahf::{
    components::transform::SolutionTransformer,
    graph::DiIntGraph,
    problems::{
        algorithm_design::{
            performance, AlgorithmDesignProblem, EvaluationStatistics, GlobalEvaluationStatistics,
            MetaheuristicIslandDesignEvaluator, TunedParams,
        },
        DirectedGraphProblem,
    },
};
use irace_rs::scenario::{Scenario, Verbosity};
use itertools::izip;
use log::info;
use mahf::{
    components::archive::ElitistArchive,
    conditions,
    lens::common::{BestObjectiveValueLens, ObjectiveValuesLens},
    population::IntoIndividuals,
    problems::{Evaluate, KnownOptimumProblem, LimitedVectorProblem},
    ExecResult, Individual, Random, SingleObjectiveProblem, State,
};

use crate::{
    cli::TrainArgs,
    config::TrainingParams,
    heuristic::grahf,
    islands::{self, island_builders, ISLAND_DE, ISLAND_ES, ISLAND_SA},
    migrations::migration_builders,
    problems::DimensionAwareDomain,
};

/// Maximum number of IRACE jobs spawned for a single candidate graph.
///
/// IRACE keeps one R process per job, so the cap bounds memory use on large machines.
const MAX_TUNING_JOBS: usize = 10;

/// Builds the seeded initial population of island graphs.
///
/// The population mixes singletons with star, wheel and complete topologies so the search
/// starts from structurally diverse candidates.
///
/// # Arguments
///
/// * `problem` - Supplies the valid node and edge weights.
/// * `rng` - Random number generator used for the topologies and weights.
///
/// # Returns
///
/// The initial individuals.
pub fn initial_population<P: SingleObjectiveProblem + KnownOptimumProblem>(
    problem: &AlgorithmDesignProblem<P>,
    rng: &mut Random,
) -> Vec<Individual<AlgorithmDesignProblem<P>>> {
    let node_types = problem.node_types();
    let edge_types = problem.edge_types();

    // The groups are built in order - singletons, then all stars, then all wheels, then all
    // complete graphs - because that order also fixes how the seeded `rng` is consumed.
    [
        // Singletons seed the search with the standalone search islands.
        vec![
            DiIntGraph::gen_singleton(&[ISLAND_DE], rng),
            DiIntGraph::gen_singleton(&[ISLAND_ES], rng),
            DiIntGraph::gen_singleton(&[ISLAND_SA], rng),
        ],
        (4..8)
            .map(|n| DiIntGraph::gen_star(n, &node_types, &edge_types, rng))
            .collect(),
        (4..8)
            .map(|n| DiIntGraph::gen_wheel(n, &node_types, &edge_types, rng))
            .collect(),
        (4..8)
            .map(|n| DiIntGraph::gen_complete(n, &node_types, &edge_types, rng))
            .collect(),
    ]
    .into_iter()
    .flatten()
    .into_individuals()
}

/// Runs the GRAHF structure search.
///
/// # Arguments
///
/// * `args` - The seed and output directory of the search.
/// * `params` - The GRAHF search's tuning parameters, loaded from `params_training.conf`.
/// * `instances` - The problem instances to optimize on.
/// * `evaluators` - One evaluator per instance.
/// * `run_dir` - Directory receiving the IRACE working files.
/// * `transformer` - Resizes migrants between islands of different dimensions.
/// * `jobs` - Total number of worker threads available to the search.
///
/// # Returns
///
/// The constructed design problem and the final search state.
///
/// # Errors
///
/// Returns an error if the working directories cannot be created or the search fails.
#[allow(clippy::too_many_arguments)]
pub fn run<P, O>(
    args: &TrainArgs,
    params: &TrainingParams,
    instances: Vec<P>,
    evaluators: Vec<O>,
    run_dir: &Path,
    transformer: Arc<dyn SolutionTransformer<P>>,
    jobs: usize,
) -> ExecResult<(
    AlgorithmDesignProblem<P>,
    State<'static, AlgorithmDesignProblem<P>>,
)>
where
    P: SingleObjectiveProblem
        + LimitedVectorProblem<Element = f64>
        + KnownOptimumProblem
        + DimensionAwareDomain
        + Send
        + Sync,
    O: Evaluate<Problem = P> + Clone + 'static,
{
    let irace_dir = run_dir.join("irace");
    fs::create_dir_all(&irace_dir)
        .wrap_err_with(|| format!("failed to create {}", irace_dir.display()))?;
    info!("Using irace working directory {}.", irace_dir.display());

    let tuning_jobs = jobs.clamp(1, MAX_TUNING_JOBS);
    info!("Using {tuning_jobs} jobs per individual for tuning.");

    let scenario = Scenario::builder()
        .min_experiments(params.num_tuning_experiments)
        .num_jobs(tuning_jobs)
        .verbose(Verbosity::Debug)
        .exec_dir(irace_dir)
        .build();

    let problem = AlgorithmDesignProblem::builder()
        .epsilon(params.epsilon)
        .max_evaluations(params.max_evaluations)
        .num_repetitions(params.num_repetitions)
        .num_tuning_repetitions(params.num_tuning_repetitions)
        .performance_measure(Box::new(performance::MedianBestObjectiveValue::new()))
        .island_builders(island_builders(
            &params.dimensions_allowed,
            params.max_island_iterations,
            params.max_island_population,
        ))
        .dependent_island_builders([islands::archive::Builder::new::<P>(
            &params.dimensions_allowed,
            params.max_island_population,
        )
        .id()])
        .migration_builders(migration_builders())
        .instances(instances.into_iter().map(Arc::new).collect())
        .tuning_scenario(scenario)
        .build();

    let config = grahf(
        params.grahf,
        conditions::LessThanN::iterations(params.num_iterations),
    )?;

    let mut rng = Random::new(args.seed);
    let initial_population = initial_population(&problem, &mut rng);

    let population_dir = args.experiments_dir.join("r").join(args.seed.to_string());
    write_initial_population(&initial_population, &population_dir)?;

    let state = config.optimize_with(&problem, |state| {
        state.populations_mut().push(initial_population);
        state.insert(rng);
        state.insert_evaluator(MetaheuristicIslandDesignEvaluator::with_transformer(
            evaluators,
            transformer,
        ));
        state.configure_log(|config| {
            config
                .with(
                    conditions::EveryN::iterations(1),
                    ObjectiveValuesLens::entry(),
                )
                .with(
                    conditions::ChangeOf::best_solution()?,
                    BestObjectiveValueLens::entry(),
                );
            Ok(())
        })?;
        Ok(())
    })?;

    info!("Optimization completed successfully.");

    Ok((problem, state))
}

/// Writes the initial population to `dir` as `dot` and JSON files.
///
/// # Arguments
///
/// * `population` - The individuals to write.
/// * `dir` - Destination directory, created if missing.
///
/// # Errors
///
/// Returns an error if the directory or any file cannot be written.
fn write_initial_population<P: SingleObjectiveProblem + KnownOptimumProblem>(
    population: &[Individual<AlgorithmDesignProblem<P>>],
    dir: &Path,
) -> ExecResult<()> {
    fs::create_dir_all(dir).wrap_err_with(|| format!("failed to create {}", dir.display()))?;

    for (i, individual) in population.iter().enumerate() {
        let graph = individual.solution();
        graph.to_dot(dir.join(format!("random_{i}.dot")))?;

        let file = fs::File::create(dir.join(format!("random_{i}.json")))?;
        serde_json::to_writer_pretty(BufWriter::new(file), graph)?;
    }

    info!("Wrote initial population to {}.", dir.display());
    Ok(())
}

/// Writes the search summary: the elitist graphs, their parameters and the run statistics.
///
/// # Arguments
///
/// * `problem` - The design problem, used to resolve builder ids and instance names.
/// * `state` - The final search state.
/// * `dir` - Destination directory, created if missing.
///
/// # Errors
///
/// Returns an error if the directory or any artefact cannot be written.
pub fn summarize<P: SingleObjectiveProblem + KnownOptimumProblem>(
    problem: &AlgorithmDesignProblem<P>,
    state: &mut State<AlgorithmDesignProblem<P>>,
    dir: &Path,
) -> ExecResult<()> {
    fs::create_dir_all(dir).wrap_err_with(|| format!("failed to create {}", dir.display()))?;
    state.log().to_json(dir.join("run.log"))?;

    let archive = state.borrow::<ElitistArchive<AlgorithmDesignProblem<P>>>();
    for (i, elitist) in archive.elitists().iter().enumerate() {
        let params = elitist.state().borrow_value::<TunedParams>();
        let solution = elitist.solution();
        let named_solution = problem.id_graph(solution);
        let statistics = elitist.state().borrow::<EvaluationStatistics>();

        info!("Elitist {i}:\n{named_solution}");
        info!("Tuning: {params:?}");
        info!("f(x) = {}", elitist.objective().value());

        for (instance, statistics) in izip!(&problem.instances, &statistics.0) {
            match statistics {
                Some(value) => info!("{}: median best objective = {value}", instance.name()),
                None => info!("Failed all runs on {}.", instance.name()),
            }
        }

        named_solution.to_dot(dir.join(format!("elitist_{i}.dot")))?;

        let file = fs::File::create(dir.join(format!("elitist_{i}.json")))?;
        serde_json::to_writer_pretty(BufWriter::new(file), solution)?;

        let file = fs::File::create(dir.join(format!("elitist_{i}.params")))?;
        serde_json::to_writer_pretty(BufWriter::new(file), &format!("{params:?}"))?;
    }
    drop(archive);

    let statistics = state.borrow::<GlobalEvaluationStatistics>();
    let statistics_map: HashMap<_, _> = izip!(&problem.instances, &statistics.0)
        .map(|(instance, statistics)| {
            (
                instance.name(),
                HashMap::from([("median_best_objective", &statistics.data)]),
            )
        })
        .collect();

    let file = fs::File::create(dir.join("statistics.json"))?;
    serde_json::to_writer_pretty(BufWriter::new(file), &statistics_map)?;

    Ok(())
}
