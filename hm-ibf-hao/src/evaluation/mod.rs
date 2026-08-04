//! Evaluation of a trained GRAHF island graph across a range of seeds.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};

use eyre::{ensure, WrapErr};
use grahf::components::{island::MigrationTransformer, transform::SolutionTransformer};
use log::info;
use mahf::{
    conditions,
    lens::{common::BestObjectiveValueLens, ValueOf},
    logging::{extractor::EntryExtractor, log::Entry},
    prelude::*,
    problems::SingleObjectiveProblem,
    state::common as mcommon,
    ExecResult, Random, State,
};

use crate::{
    alignment::{
        output::{select_global_best_solution, OUTPUT_TRANSFORM_NOT_APPLIED},
        write_run_results, AlignmentEvaluator, HorizontalAlignment, RunMetadata,
    },
    cli::EvaluateArgs,
    config::EvaluationParams,
};

pub mod elitist;
pub mod log_series;

pub use elitist::load_elitist;
pub use log_series::{parse_log_series, write_progress_csvs, ProgressSeries};

/// Name recorded for the evaluated algorithm.
const ALGORITHM: &str = "GRAHF";

/// Logs the objective values of every island's current population.
///
/// The mean of these values is what `avg_value.csv` reports, so the entry has to see every
/// island's population rather than only the run's global best.
#[derive(Clone)]
pub struct IslandObjectiveValuesEntry;

impl<P> EntryExtractor<P> for IslandObjectiveValuesEntry
where
    P: SingleObjectiveProblem + 'static,
{
    fn extract_entry(&self, _problem: &P, state: &State<P>) -> Entry {
        let mut values = Vec::new();

        if let Ok(states) = state.try_borrow::<grahf::components::island::IslandStates<P>>() {
            for island_state in states.iter() {
                if let Ok(populations) = island_state.try_borrow::<mcommon::Populations<P>>() {
                    if let Some(population) = populations.get_current() {
                        values.extend(
                            population
                                .iter()
                                .filter_map(|individual| individual.get_objective())
                                .map(|objective| objective.value()),
                        );
                    }
                }
            }
        }

        Entry {
            name: "ObjectiveValues",
            value: Box::new(values),
        }
    }
}

/// The per-seed results of one instance.
#[derive(Debug, Default)]
struct InstanceSummary {
    /// The instance's natural dimension.
    natural_dimension: usize,
    /// The best objective value reached in each seed's run.
    best_values: Vec<f64>,
}

/// Evaluates the trained elitist on every instance for every requested seed.
///
/// One result folder is written per `(instance, seed)` pair and an aggregated summary CSV is
/// written afterwards, holding the mean and sample standard deviation per instance.
///
/// # Arguments
///
/// * `args` - Seeds and output locations.
/// * `eval_params` - The evaluate stage's tuning parameters, loaded from
///   `params_evaluation.conf`.
/// * `dimensions_allowed` - Allowed island working dimensions; see
///   [`crate::config::TrainingParams::dimensions_allowed`].
/// * `run_dir` - Directory holding the trained artefacts.
/// * `instances_dir` - Directory holding the instances to evaluate on.
/// * `max_evaluations` - Evaluation budget of a single run.
/// * `transformer` - Resizes migrants between islands of different dimensions.
///
/// # Returns
///
/// Nothing; all results are written to disk.
///
/// # Errors
///
/// Returns an error if the artefacts or instances cannot be loaded, a run fails, the exported
/// best individual is inconsistent with the run's state, or a file cannot be written.
#[allow(clippy::too_many_arguments)]
pub fn run(
    args: &EvaluateArgs,
    eval_params: &EvaluationParams,
    dimensions_allowed: &[u32],
    run_dir: &Path,
    instances_dir: &Path,
    max_evaluations: u32,
    transformer: Arc<dyn SolutionTransformer<HorizontalAlignment>>,
) -> ExecResult<()> {
    fs::create_dir_all(&args.results_dir)
        .wrap_err_with(|| format!("failed to create {}", args.results_dir.display()))?;

    let (builder_graph, params) = load_elitist::<HorizontalAlignment>(
        run_dir,
        &args.elitist,
        dimensions_allowed,
        eval_params.max_iterations,
        eval_params.max_population_size,
    )?;

    info!("Loading instances from {}", instances_dir.display());
    let instances = HorizontalAlignment::load_instances(instances_dir, dimensions_allowed)?;
    info!("Loaded {} instances.", instances.len());
    info!("Allowed island dimensions: {dimensions_allowed:?}");

    let builder = builder_graph.into_builder(conditions::LessThanN::evaluations(max_evaluations));
    let config = builder(params)?;

    let mut summaries: BTreeMap<String, InstanceSummary> = BTreeMap::new();

    for seed in args.seeds() {
        info!("=== SEED {seed} ===");

        for instance in &instances {
            let best_value = evaluate_instance(
                &config,
                instance,
                seed,
                &args.results_dir,
                transformer.clone(),
                eval_params.best_value_tolerance,
                dimensions_allowed,
            )?;

            let summary = summaries.entry(instance.name.clone()).or_default();
            summary.natural_dimension = instance.natural_dimension;
            summary.best_values.push(best_value);
        }
    }

    write_summary_csv(&args.summary_csv, &summaries)?;
    info!("Wrote summary to {}", args.summary_csv.display());

    Ok(())
}

/// Runs the configured metaheuristic once and writes the run's artefacts.
///
/// # Arguments
///
/// * `config` - The metaheuristic built from the trained elitist.
/// * `instance` - The instance to solve.
/// * `seed` - The run's random seed.
/// * `results_dir` - Directory receiving the run folder.
/// * `transformer` - Resizes migrants between islands of different dimensions.
/// * `best_value_tolerance` - Tolerance when cross-checking the exported best value against
///   the run's state.
/// * `dimensions_allowed` - Allowed island working dimensions; see
///   [`crate::config::TrainingParams::dimensions_allowed`].
///
/// # Returns
///
/// The best objective value reached by the run.
///
/// # Errors
///
/// Returns an error if the run fails, the exported best individual disagrees with the run's
/// state, its dimension is not an allowed island dimension, or an artefact cannot be written.
#[allow(clippy::too_many_arguments)]
fn evaluate_instance(
    config: &Configuration<HorizontalAlignment>,
    instance: &HorizontalAlignment,
    seed: u64,
    results_dir: &Path,
    transformer: Arc<dyn SolutionTransformer<HorizontalAlignment>>,
    best_value_tolerance: f64,
    dimensions_allowed: &[u32],
) -> ExecResult<f64> {
    let instance_name = instance.name.clone();
    info!("Starting run on {instance_name} (seed={seed})");

    let state = config.optimize_with(instance, |state| {
        state.insert_evaluator(AlignmentEvaluator);
        state.insert(Random::new(seed));
        state.insert(MigrationTransformer(transformer.clone()));
        state.configure_log(|cfg| {
            cfg.with(
                conditions::EveryN::iterations(1),
                BestObjectiveValueLens::entry(),
            );
            cfg.with(
                conditions::EveryN::iterations(1),
                ValueOf::<mcommon::Evaluations>::entry(),
            );
            cfg.with(
                conditions::EveryN::iterations(1),
                Box::new(IslandObjectiveValuesEntry),
            );
            Ok(())
        })?;
        Ok(())
    })?;

    let state_best_value = state
        .best_objective_value()
        .ok_or_else(|| eyre::eyre!("run on {instance_name} (seed={seed}) produced no best value"))?
        .value();

    let global_best = select_global_best_solution(&state).wrap_err_with(|| {
        format!("failed to select global best individual for {instance_name} (seed={seed})")
    })?;

    ensure!(
        (state_best_value - global_best.best_value).abs() <= best_value_tolerance,
        "state best value {state_best_value} does not match global best individual value {} \
         for {instance_name} (seed={seed})",
        global_best.best_value,
    );
    ensure!(
        dimensions_allowed.contains(&(global_best.solution_dim as u32)),
        "global best solution for {instance_name} (seed={seed}) has unsupported dimension {}; \
         allowed {:?}",
        global_best.solution_dim,
        dimensions_allowed,
    );

    info!(
        "f(x) global_best={:.6} solution_dim={} natural_dimension={}",
        global_best.best_value, global_best.solution_dim, instance.natural_dimension,
    );

    let metadata = RunMetadata {
        problem: &instance_name,
        natural_dimension: instance.natural_dimension,
        backbone_length: instance.backbone_total_length,
        solution_dim: global_best.solution_dim,
        output_transform_method: OUTPUT_TRANSFORM_NOT_APPLIED,
        solution: &global_best.solution,
    };

    let run_dir = write_run_results(
        results_dir,
        &metadata,
        ALGORITHM,
        seed,
        u64::from(state.evaluations()),
        global_best.best_value,
    )?;
    // `Log::to_json` is the only public serialization path, so the log is round-tripped
    // through a file inside the run folder before the progress series is extracted.
    let log_path = run_dir.join("run.log");
    state.log().to_json(&log_path)?;
    let log_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&log_path)
            .wrap_err_with(|| format!("failed to read {}", log_path.display()))?,
    )
    .wrap_err_with(|| format!("failed to parse {}", log_path.display()))?;
    write_progress_csvs(&run_dir, &parse_log_series(&log_json)?)?;

    info!("Stored results in {}", run_dir.display());

    Ok(global_best.best_value)
}

/// Writes the aggregated evaluation summary.
///
/// # Arguments
///
/// * `path` - Destination CSV file.
/// * `summaries` - Per-instance best values, one entry per evaluated seed.
///
/// # Errors
///
/// Returns an error if the file cannot be created or written.
fn write_summary_csv(path: &Path, summaries: &BTreeMap<String, InstanceSummary>) -> ExecResult<()> {
    let file =
        File::create(path).wrap_err_with(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "Algorithm,Instance,NaturalDimension,NSeeds,Mean,Std"
    )?;
    for (instance, summary) in summaries {
        let n = summary.best_values.len();
        let mean = summary.best_values.iter().sum::<f64>() / n as f64;
        // Sample standard deviation; a single seed carries no spread information.
        let std = if n >= 2 {
            let variance = summary
                .best_values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (n - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        writeln!(
            writer,
            "{ALGORITHM},{instance},{},{n},{mean:.6},{std:.6}",
            summary.natural_dimension
        )?;
    }
    writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a summary of the given per-seed best values.
    fn summary(natural_dimension: usize, best_values: &[f64]) -> InstanceSummary {
        InstanceSummary {
            natural_dimension,
            best_values: best_values.to_vec(),
        }
    }

    #[test]
    fn the_summary_holds_one_row_per_instance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results_hao.csv");
        let summaries = BTreeMap::from([
            ("alps".to_string(), summary(60, &[1.0, 3.0])),
            ("dolomites".to_string(), summary(53, &[10.0])),
        ]);

        write_summary_csv(&path, &summaries).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<_> = content.lines().collect();
        assert_eq!(
            lines[0],
            "Algorithm,Instance,NaturalDimension,NSeeds,Mean,Std"
        );
        assert_eq!(lines[1], "GRAHF,alps,60,2,2.000000,1.414214");
        assert_eq!(
            lines[2], "GRAHF,dolomites,53,1,10.000000,0.000000",
            "n=1 has no spread"
        );
    }

    #[test]
    fn the_summary_rows_are_ordered_by_instance_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results_hao.csv");
        let summaries = BTreeMap::from([
            ("zeta".to_string(), summary(1, &[1.0])),
            ("alpha".to_string(), summary(1, &[1.0])),
        ]);

        write_summary_csv(&path, &summaries).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let names: Vec<_> = content
            .lines()
            .skip(1)
            .map(|line| line.split(',').nth(1).unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["alpha", "zeta"]);
    }

    #[test]
    fn an_empty_summary_still_has_a_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results_hao.csv");

        write_summary_csv(&path, &BTreeMap::new()).unwrap();

        assert_eq!(
            fs::read_to_string(&path).unwrap().trim(),
            "Algorithm,Instance,NaturalDimension,NSeeds,Mean,Std"
        );
    }

    #[test]
    fn writing_the_summary_to_a_missing_directory_reports_the_path() {
        let error =
            write_summary_csv(Path::new("no/such/dir/out.csv"), &BTreeMap::new()).unwrap_err();

        assert!(error.to_string().contains("no/such/dir/out.csv"), "{error}");
    }
}
