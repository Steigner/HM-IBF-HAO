//! End-to-end smoke test of the evaluate stage.
//!
//! `tests/pipeline.rs` drives the benchmark's pieces in isolation; this file runs the whole
//! `evaluate` stage the way the binary does — load a trained elitist from disk, rebuild the
//! island graph, replay it on an instance and write the run artefacts — so that a break in
//! the wiring between those pieces fails the verification gate rather than the first real
//! experiment.
//!
//! The training stage is deliberately not covered here: it shells out to R/IRACE, which is
//! only present in the Nix image. `scripts/smoke.sh` covers preprocess → train → evaluate
//! there; everything that runs without IRACE is covered here.

use std::{fs, path::Path, sync::Arc};

use grahf::components::transform::SolutionTransformer;
use grahf_hao::{
    alignment::{AlignmentConfig, BackboneConfig, HorizontalAlignment},
    cli::EvaluateArgs,
    config::EvaluationParams,
    evaluation,
    islands::transforms::OffsetResampleTransformer,
};
use ndarray::Array2;

/// Working dimensions the smoke run's islands may use; small enough to stay fast.
///
/// Deliberately unrelated to the shipped `params_training.conf`, so a change to the
/// benchmark's own dimension set cannot alter what this test exercises.
const SMOKE_DIMENSIONS: [u32; 2] = [3, 5];

/// Evaluation budget of the smoke run. Large enough to finish an island iteration, small
/// enough that the whole test stays well under a second.
const MAX_EVALUATIONS: u32 = 60;

/// Number of backbone points of the synthetic instance.
const BACKBONE_POINTS: usize = 20;

/// A trained elitist holding a single differential evolution island and no migration edge.
///
/// Node weight `0` is `ISLAND_DE`; a single node keeps the run deterministic in structure
/// while still exercising the full rebuild path.
const ELITIST_GRAPH: &str =
    r#"{"graph":{"nodes":[0],"node_holes":[],"edge_property":"directed","edges":[]}}"#;

/// The IRACE parameters the training stage would have stored for that graph.
const ELITIST_PARAMS: &str = r#"{
    "island": {
        "0": {
            "dimension": 5,
            "population_size": 4,
            "iterations": 2,
            "y": 1,
            "selection": "rand",
            "crossover": "binomial",
            "pc": 0.9,
            "f": 0.5
        }
    },
    "migration": {}
}"#;

/// Builds a straight, flat instance on a level heightmap.
///
/// # Returns
///
/// A config whose backbone runs down the middle of a `20 x 20` terrain.
fn smoke_config() -> AlignmentConfig {
    let points: Vec<Vec<f64>> = (0..BACKBONE_POINTS).map(|i| vec![i as f64, 10.0]).collect();
    let distances: Vec<f64> = (0..BACKBONE_POINTS).map(|i| i as f64).collect();

    AlignmentConfig {
        name: "smoke".to_string(),
        heightmap_shape: vec![BACKBONE_POINTS, BACKBONE_POINTS],
        start: vec![0, 10],
        goal: vec![(BACKBONE_POINTS - 1) as i32, 10],
        path_astar: (0..BACKBONE_POINTS)
            .map(|i| vec![i as i32, 10])
            .collect::<Vec<_>>(),
        path_simplified: vec![vec![0, 10], vec![(BACKBONE_POINTS - 1) as i32, 10]],
        backbone: BackboneConfig {
            cumulative_distances: distances,
            total_length: (BACKBONE_POINTS - 1) as f64,
            normals: vec![vec![0.0, 1.0]; BACKBONE_POINTS],
            offset_bounds: vec![vec![-5.0, 5.0]; BACKBONE_POINTS],
            points,
        },
        epsilon: 1.0,
        cutting_plane_factor: 1.0,
        tunnel_factor: 5.0,
        gradient_factor: 2.0,
        curvature_radius: 100.0,
        gradient_change_limit: 0.08,
        height_limit: 800.0,
        tau: 0.4,
    }
}

/// Writes the synthetic instance and the trained elitist a smoke run needs.
///
/// # Arguments
///
/// * `instances_dir` - Directory receiving `smoke_config.json` and `smoke_heightmap.npy`.
/// * `run_dir` - Directory receiving `elitist_0.json` and `elitist_0.params`.
fn write_fixtures(instances_dir: &Path, run_dir: &Path) {
    fs::create_dir_all(instances_dir).unwrap();
    fs::create_dir_all(run_dir).unwrap();

    fs::write(
        instances_dir.join("smoke_config.json"),
        serde_json::to_string_pretty(&smoke_config()).unwrap(),
    )
    .unwrap();
    ndarray_npy::write_npy(
        instances_dir.join("smoke_heightmap.npy"),
        &Array2::<f32>::zeros((BACKBONE_POINTS, BACKBONE_POINTS)),
    )
    .unwrap();

    fs::write(run_dir.join("elitist_0.json"), ELITIST_GRAPH).unwrap();
    fs::write(run_dir.join("elitist_0.params"), ELITIST_PARAMS).unwrap();
}

/// Returns the evaluate stage's arguments for a single-seed smoke run.
///
/// # Arguments
///
/// * `root` - Temporary directory the results are written below.
fn smoke_args(root: &Path) -> EvaluateArgs {
    EvaluateArgs {
        first_seed: 1,
        num_seeds: 1,
        elitist: "elitist_0".to_string(),
        results_dir: root.join("results"),
        summary_csv: root.join("results_hao.csv"),
    }
}

/// Runs the evaluate stage over the fixtures and returns the run's result directory.
///
/// # Arguments
///
/// * `root` - Temporary directory holding every input and output of the run.
fn run_evaluate(root: &Path) -> std::path::PathBuf {
    let instances_dir = root.join("instances");
    let run_dir = root.join("hao_run");
    write_fixtures(&instances_dir, &run_dir);

    let args = smoke_args(root);
    let params = EvaluationParams {
        max_iterations: 100,
        max_population_size: 128,
        best_value_tolerance: 1e-9,
    };
    let transformer: Arc<dyn SolutionTransformer<HorizontalAlignment>> =
        Arc::new(OffsetResampleTransformer::new());

    evaluation::run(
        &args,
        &params,
        &SMOKE_DIMENSIONS,
        &run_dir,
        &instances_dir,
        MAX_EVALUATIONS,
        transformer,
    )
    .expect("the evaluate stage must complete on the smoke fixtures");

    let mut runs: Vec<_> = fs::read_dir(&args.results_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .collect();
    assert_eq!(runs.len(), 1, "one instance times one seed");
    runs.pop().unwrap()
}

#[test]
fn the_evaluate_stage_writes_every_documented_artefact() {
    let root = tempfile::tempdir().unwrap();

    let run = run_evaluate(root.path());

    for artefact in ["results.json", "best_value.csv", "avg_value.csv", "run.log"] {
        assert!(run.join(artefact).is_file(), "missing {artefact}");
    }
    assert!(root.path().join("results_hao.csv").is_file());
}

#[test]
fn the_run_folder_is_named_after_the_instance_the_algorithm_and_the_seed() {
    let root = tempfile::tempdir().unwrap();

    let run = run_evaluate(root.path());

    let name = run.file_name().unwrap().to_str().unwrap();
    assert!(name.starts_with("smoke_GRAHF_seed1_"), "{name}");
}

#[test]
fn the_exported_payload_describes_the_run_it_came_from() {
    let root = tempfile::tempdir().unwrap();

    let run = run_evaluate(root.path());
    let payload: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(run.join("results.json")).unwrap()).unwrap();

    assert_eq!(payload["algorithm"], "GRAHF");
    assert_eq!(payload["problem"], "smoke");
    assert_eq!(payload["seed"].as_u64().unwrap(), 1);

    // The exported dimension must be one the islands were allowed to run at: that is the
    // end-to-end proof that the transform chain never changed a solution's length.
    let solution_dim = payload["solution_dim"].as_u64().unwrap() as u32;
    assert!(
        SMOKE_DIMENSIONS.contains(&solution_dim),
        "exported dimension {solution_dim} is not in {SMOKE_DIMENSIONS:?}"
    );
    assert_eq!(
        payload["x"].as_array().unwrap().len(),
        solution_dim as usize
    );
}

#[test]
fn the_exported_value_is_reproducible_from_the_exported_solution() {
    let root = tempfile::tempdir().unwrap();

    let run = run_evaluate(root.path());
    let payload: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(run.join("results.json")).unwrap()).unwrap();

    let instance = HorizontalAlignment::from_config(
        smoke_config(),
        Array2::zeros((BACKBONE_POINTS, BACKBONE_POINTS)),
        &SMOKE_DIMENSIONS,
    )
    .unwrap();
    let solution: Vec<f64> = payload["x"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();

    let reported = payload["best_value"].as_f64().unwrap();
    assert!(
        (instance.evaluate_solution(&solution) - reported).abs() < 1e-9,
        "recomputing f(x) from the export must reproduce {reported}"
    );
}

#[test]
fn the_run_stays_inside_its_evaluation_budget() {
    let root = tempfile::tempdir().unwrap();

    let run = run_evaluate(root.path());
    let payload: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(run.join("results.json")).unwrap()).unwrap();

    let evaluations = payload["n_evals"].as_u64().unwrap();
    assert!(evaluations > 0, "the run must have evaluated something");
    // The budget is a stopping condition checked between components, so a single island
    // iteration may overshoot it; it must not run away from it.
    assert!(
        evaluations < u64::from(MAX_EVALUATIONS) * 2,
        "{evaluations} evaluations against a budget of {MAX_EVALUATIONS}"
    );
}

#[test]
fn the_progress_series_are_written_with_their_headers() {
    let root = tempfile::tempdir().unwrap();

    let run = run_evaluate(root.path());

    let best = fs::read_to_string(run.join("best_value.csv")).unwrap();
    assert_eq!(best.lines().next(), Some("nfes,best_value"));
    assert!(
        best.lines().count() > 1,
        "the best series must have samples"
    );

    let average = fs::read_to_string(run.join("avg_value.csv")).unwrap();
    assert_eq!(average.lines().next(), Some("nfes,avg_value"));
}

#[test]
fn the_summary_holds_one_row_for_the_evaluated_instance() {
    let root = tempfile::tempdir().unwrap();

    run_evaluate(root.path());

    let summary = fs::read_to_string(root.path().join("results_hao.csv")).unwrap();
    let lines: Vec<_> = summary.lines().collect();

    assert_eq!(
        lines[0],
        "Algorithm,Instance,NaturalDimension,NSeeds,Mean,Std"
    );
    assert_eq!(lines.len(), 2, "one instance");
    assert!(lines[1].starts_with("GRAHF,smoke,"), "{}", lines[1]);
}
