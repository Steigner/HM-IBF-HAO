//! Loading of the TOML settings files holding the benchmark's algorithm tuning parameters.
//!
//! `params_training.conf` and `params_evaluation.conf` sit at the repository root, next to
//! `rustfmt.toml` and `nix.conf`, so the search and rebuild parameters can be edited without
//! recompiling. Run identity (seeds) and file locations stay CLI flags; see `cli.rs`.

use std::path::Path;

use eyre::{ensure, WrapErr};
use serde::{de::DeserializeOwned, Deserialize};

use crate::heuristic::GrahfParameters;

/// Default location of [`TrainingParams`], relative to the `hm-ibf-hao` working directory.
pub const DEFAULT_TRAINING_PARAMS: &str = "../params_training.conf";

/// Default location of [`EvaluationParams`], relative to the `hm-ibf-hao` working directory.
pub const DEFAULT_EVALUATION_PARAMS: &str = "../params_evaluation.conf";

/// Tuning parameters of the GRAHF structure search, loaded from `params_training.conf`.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingParams {
    /// Termination tolerance against the known optimum of the inner problem.
    pub epsilon: f64,
    /// Evaluation budget of a single inner metaheuristic run; shared with the evaluate stage.
    pub max_evaluations: u32,
    /// Repetitions per instance when scoring a candidate graph.
    pub num_repetitions: u32,
    /// Repetitions per instance during IRACE tuning.
    pub num_tuning_repetitions: u32,
    /// Minimum number of IRACE experiments per candidate graph.
    pub num_tuning_experiments: u32,
    /// Generations of the outer structure search.
    pub num_iterations: u32,
    /// Upper bound IRACE may assign to an island's iteration count.
    pub max_island_iterations: u32,
    /// Upper bound IRACE may assign to an island's population size.
    pub max_island_population: u32,
    /// Hyper-parameters of the outer graph-level genetic algorithm.
    pub grahf: GrahfParameters,
    /// Allowed island working dimensions in offsets along the backbone, strictly increasing.
    ///
    /// The set is a *recommendation derived from the Douglas-Peucker compression* of each
    /// instance's raw route: the compression estimates how many interior inflection points a
    /// route really has (its `natural_dimension`), and the allowed set is assembled from those
    /// per-instance estimates. The shipped `[53, 60, 68]` are the natural dimensions of the
    /// three benchmark maps. Validated by [`TrainingParams::load`].
    ///
    /// The list must stay strictly increasing: `HorizontalAlignment::dimension()` and
    /// `domain()` both read its last entry as the maximum, so an unsorted list would size the
    /// declared search space below the largest island and break its initialization.
    pub dimensions_allowed: Vec<u32>,
}

impl TrainingParams {
    /// Loads the training tuning parameters from a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to `params_training.conf`.
    ///
    /// # Returns
    ///
    /// The parsed parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, does not hold a valid configuration, or
    /// `dimensions_allowed` is empty, not strictly increasing, or contains a zero.
    pub fn load(path: &Path) -> eyre::Result<Self> {
        let params: Self = load_toml(path)?;
        validate_dimensions_allowed(&params.dimensions_allowed)
            .wrap_err_with(|| format!("invalid dimensions_allowed in {}", path.display()))?;
        Ok(params)
    }
}

/// Validates that `dimensions` is a non-empty, strictly increasing list of positive values.
///
/// # Arguments
///
/// * `dimensions` - The candidate `dimensions_allowed` list.
///
/// # Errors
///
/// Returns an error describing the first violated invariant.
fn validate_dimensions_allowed(dimensions: &[u32]) -> eyre::Result<()> {
    ensure!(
        !dimensions.is_empty(),
        "dimensions_allowed must not be empty"
    );
    ensure!(
        dimensions.windows(2).all(|w| w[0] < w[1]),
        "dimensions_allowed must be strictly increasing: {dimensions:?}"
    );
    ensure!(
        dimensions.iter().all(|&d| d > 0),
        "every entry of dimensions_allowed must be positive: {dimensions:?}"
    );
    Ok(())
}

/// Tuning parameters of the evaluate stage, loaded from `params_evaluation.conf`.
///
/// Evaluation replays the trained graph's stored, IRACE-tuned parameters instead of
/// sampling new ones, so `max_iterations` and `max_population_size` only shape the
/// parameter space and never change a run's outcome; they live in configuration rather
/// than as CLI flags to avoid suggesting otherwise.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct EvaluationParams {
    /// Upper bound IRACE may assign to an island's iteration count when rebuilding the
    /// trained graph.
    pub max_iterations: u32,
    /// Island population bound used when rebuilding the trained graph.
    pub max_population_size: u32,
    /// Tolerance when cross-checking the exported best value against the run's state.
    pub best_value_tolerance: f64,
}

impl EvaluationParams {
    /// Loads the evaluation tuning parameters from a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to `params_evaluation.conf`.
    ///
    /// # Returns
    ///
    /// The parsed parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or does not hold a valid configuration.
    pub fn load(path: &Path) -> eyre::Result<Self> {
        load_toml(path)
    }
}

/// Reads and parses a TOML settings file.
///
/// # Arguments
///
/// * `path` - Path to the TOML file.
///
/// # Returns
///
/// The deserialized value.
///
/// # Errors
///
/// Returns an error if the file cannot be read or is not valid TOML for `T`.
fn load_toml<T: DeserializeOwned>(path: &Path) -> eyre::Result<T> {
    let content = std::fs::read_to_string(path)
        .wrap_err_with(|| format!("failed to read {}", path.display()))?;
    toml::from_str(&content).wrap_err_with(|| format!("failed to parse {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a valid `params_training.conf` whose `dimensions_allowed` is `dimensions`.
    fn write_training_params_with_dimensions(dir: &Path, dimensions: &str) -> std::path::PathBuf {
        let path = dir.join("params_training.conf");
        std::fs::write(
            &path,
            format!(
                r#"
                epsilon = 1e-8
                max_evaluations = 100_000
                num_repetitions = 2
                num_tuning_repetitions = 3
                num_tuning_experiments = 10
                num_iterations = 4
                max_island_iterations = 30
                max_island_population = 15
                dimensions_allowed = {dimensions}

                [grahf]
                max_initial_nodes = 3
                initial_edge_p = 0.3
                population_size = 5
                tournament_size = 2
                archive_size = 1
                elitist_freq = 2
                pc = 0.6
                rm_node = 0.12
                rm_edge = 0.15
                rm_node_weight = 0.08
                rm_edge_weight = 0.2
                "#
            ),
        )
        .unwrap();
        path
    }

    #[test]
    fn training_params_parse_from_the_documented_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_training_params_with_dimensions(dir.path(), "[53, 60, 68]");

        let params = TrainingParams::load(&path).unwrap();

        assert_eq!(params.max_evaluations, 100_000);
        assert_eq!(params.num_iterations, 4);
        assert_eq!(params.grahf.population_size, 5);
        assert_eq!(params.dimensions_allowed, vec![53, 60, 68]);
    }

    #[test]
    fn evaluation_params_parse_from_the_documented_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("params_evaluation.conf");
        std::fs::write(
            &path,
            "max_iterations = 100\nmax_population_size = 128\nbest_value_tolerance = 1e-9\n",
        )
        .unwrap();

        let params = EvaluationParams::load(&path).unwrap();

        assert_eq!(params.max_iterations, 100);
        assert_eq!(params.max_population_size, 128);
        assert_eq!(params.best_value_tolerance, 1e-9);
    }

    #[test]
    fn a_missing_training_params_file_reports_its_path() {
        let error = TrainingParams::load(Path::new("definitely/missing.conf")).unwrap_err();

        assert!(
            error.to_string().contains("definitely/missing.conf"),
            "{error}"
        );
    }

    #[test]
    fn an_incomplete_training_params_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("params_training.conf");
        std::fs::write(&path, "epsilon = 1e-8\n").unwrap();

        assert!(TrainingParams::load(&path).is_err());
    }

    #[test]
    fn an_empty_dimensions_allowed_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_training_params_with_dimensions(dir.path(), "[]");

        let error = TrainingParams::load(&path).unwrap_err();

        assert!(
            error
                .chain()
                .any(|cause| cause.to_string().contains("must not be empty")),
            "{error:?}"
        );
    }

    #[test]
    fn a_non_increasing_dimensions_allowed_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_training_params_with_dimensions(dir.path(), "[60, 53, 68]");

        let error = TrainingParams::load(&path).unwrap_err();

        assert!(
            error
                .chain()
                .any(|cause| cause.to_string().contains("strictly increasing")),
            "{error:?}"
        );
    }

    #[test]
    fn a_zero_dimension_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_training_params_with_dimensions(dir.path(), "[0, 53]");

        let error = TrainingParams::load(&path).unwrap_err();

        assert!(
            error
                .chain()
                .any(|cause| cause.to_string().contains("must be positive")),
            "{error:?}"
        );
    }

    #[test]
    fn the_repository_root_training_params_file_is_valid() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_TRAINING_PARAMS);

        let params = TrainingParams::load(&path).unwrap();

        assert_eq!(
            params.dimensions_allowed,
            vec![53, 60, 68],
            "the shipped set is the Douglas-Peucker recommendation of the three benchmark maps"
        );
    }

    #[test]
    fn the_repository_root_evaluation_params_file_is_valid() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_EVALUATION_PARAMS);

        EvaluationParams::load(&path).unwrap();
    }
}
