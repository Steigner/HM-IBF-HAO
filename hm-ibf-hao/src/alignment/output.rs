//! Result export for a single evaluation run.
//!
//! Each run produces a folder named `<instance>_<algorithm>_seed<seed>_<tag>` holding a
//! `results.json` payload and a `best_value.csv` progress series.

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use eyre::{bail, ensure, WrapErr};
use mahf::{ExecResult, State};

use super::problem::HorizontalAlignment;

/// Version of the `results.json` payload written by this module.
pub const OUTPUT_EXPORT_SCHEMA_VERSION: u32 = 1;

/// Marks results whose working dimension is chosen by the islands rather than fixed upfront.
pub const OUTPUT_ADAPTIVE_MODE: &str = "dimension_agnostic_adaptive_output";

/// Marks that the exported solution is the run's global best individual.
pub const OUTPUT_BEST_SCOPE: &str = "global_best_individual";

/// Marks that no output projection was applied to the exported solution.
pub const OUTPUT_TRANSFORM_NOT_APPLIED: &str = "none";

/// FNV-1a offset basis, used for the deterministic run tag.
const FNV_OFFSET_BASIS: u32 = 2_166_136_261;

/// FNV-1a prime, used for the deterministic run tag.
const FNV_PRIME: u32 = 16_777_619;

/// Describes one evaluation run for the exported payload.
pub struct RunMetadata<'a> {
    /// The instance name.
    pub problem: &'a str,
    /// The instance's natural dimension, i.e. its geometric complexity.
    pub natural_dimension: usize,
    /// Arc length of the instance's backbone.
    pub backbone_length: f64,
    /// Length of the exported offset vector, i.e. the winning island's dimension.
    pub solution_dim: usize,
    /// Name of the output projection applied, or [`OUTPUT_TRANSFORM_NOT_APPLIED`].
    pub output_transform_method: &'a str,
    /// The exported offsets.
    pub solution: &'a [f64],
}

/// The global best individual of a finished run.
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalBestSolution {
    /// Length of the offset vector.
    pub solution_dim: usize,
    /// The offsets of the best individual.
    pub solution: Vec<f64>,
    /// The objective value the offsets produced.
    pub best_value: f64,
}

/// Extracts the global best individual from a finished run's state.
///
/// # Arguments
///
/// * `state` - The final state of the run.
///
/// # Returns
///
/// The best individual's dimension, offsets and objective value.
///
/// # Errors
///
/// Returns an error if the run produced no best individual, its solution is empty, or its
/// objective value is not finite.
pub fn select_global_best_solution(
    state: &State<HorizontalAlignment>,
) -> ExecResult<GlobalBestSolution> {
    let Some(best_individual) = state.best_individual() else {
        bail!("missing global best individual in final GRAHF state");
    };

    let solution = best_individual.solution().clone();
    ensure!(
        !solution.is_empty(),
        "the global best solution carries no offsets"
    );

    let best_value = best_individual.objective().value();
    ensure!(
        best_value.is_finite(),
        "global best objective value is not finite: {best_value}"
    );

    Ok(GlobalBestSolution {
        solution_dim: solution.len(),
        solution,
        best_value,
    })
}

/// Writes the `results.json` payload and the single-point `best_value.csv` of one run.
///
/// The progress CSVs written later by [`crate::evaluation::write_progress_csvs`] overwrite
/// `best_value.csv` with the full series; this initial write guarantees the file exists even
/// when the run log carries no entries.
///
/// # Arguments
///
/// * `root` - Directory receiving the run folder.
/// * `metadata` - Description of the run.
/// * `algo` - The algorithm name recorded in the payload and the folder name.
/// * `seed` - The run's seed.
/// * `n_evals` - The number of objective evaluations the run consumed.
/// * `best_value` - The objective value of the exported solution.
///
/// # Returns
///
/// The created run folder.
///
/// # Errors
///
/// Returns an error if the folder or any file cannot be created or written.
pub fn write_run_results(
    root: &Path,
    metadata: &RunMetadata<'_>,
    algo: &str,
    seed: u64,
    n_evals: u64,
    best_value: f64,
) -> ExecResult<PathBuf> {
    let tag = compute_hash(metadata.problem, algo, seed);
    let dir = root.join(format!("{}_{algo}_seed{seed}_{tag}", metadata.problem));
    fs::create_dir_all(&dir).wrap_err_with(|| format!("failed to create {}", dir.display()))?;

    let mut best_file = File::create(dir.join("best_value.csv"))
        .wrap_err_with(|| format!("failed to create best_value.csv in {}", dir.display()))?;
    writeln!(best_file, "nfes,best_value")?;
    writeln!(best_file, "{n_evals},{best_value}")?;

    let payload = serde_json::json!({
        "algorithm": algo,
        "seed": seed,
        "problem": metadata.problem,
        "natural_dimension": metadata.natural_dimension,
        "backbone_length": metadata.backbone_length,
        "best_value": best_value,
        "n_evals": n_evals,
        "export_schema_version": OUTPUT_EXPORT_SCHEMA_VERSION,
        "mode": OUTPUT_ADAPTIVE_MODE,
        "output_best_scope": OUTPUT_BEST_SCOPE,
        "output_transform_method": metadata.output_transform_method,
        "solution_dim": metadata.solution_dim,
        "x": metadata.solution,
    });

    let file = File::create(dir.join("results.json"))
        .wrap_err_with(|| format!("failed to create results.json in {}", dir.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(file), &payload)?;

    Ok(dir)
}

/// Returns a short deterministic tag identifying a run folder.
///
/// # Arguments
///
/// * `run_label` - The instance name.
/// * `algo` - The algorithm name.
/// * `seed` - The run's seed.
///
/// # Returns
///
/// An eight-character lowercase hexadecimal FNV-1a digest.
fn compute_hash(run_label: &str, algo: &str, seed: u64) -> String {
    let input = format!("{run_label}_{algo}_{seed}");
    let mut hash = FNV_OFFSET_BASIS;
    for byte in input.bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds metadata for an exported offset vector.
    fn metadata(solution: &[f64]) -> RunMetadata<'_> {
        RunMetadata {
            problem: "alps_eps1.0",
            natural_dimension: 60,
            backbone_length: 435.4,
            solution_dim: solution.len(),
            output_transform_method: OUTPUT_TRANSFORM_NOT_APPLIED,
            solution,
        }
    }

    #[test]
    fn the_run_tag_is_deterministic_and_seed_dependent() {
        assert_eq!(
            compute_hash("alps_eps1.0", "GRAHF", 42),
            compute_hash("alps_eps1.0", "GRAHF", 42)
        );
        assert_ne!(
            compute_hash("alps_eps1.0", "GRAHF", 42),
            compute_hash("alps_eps1.0", "GRAHF", 43)
        );
        assert_eq!(compute_hash("a", "b", 1).len(), 8);
    }

    #[test]
    fn the_run_folder_follows_the_documented_naming_scheme() {
        let dir = tempfile::tempdir().unwrap();
        let solution = vec![0.25; 13];

        let run_dir =
            write_run_results(dir.path(), &metadata(&solution), "GRAHF", 42, 1000, 1.5).unwrap();

        let name = run_dir.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("alps_eps1.0_GRAHF_seed42_"), "{name}");
        assert_eq!(name.rsplit('_').next().unwrap().len(), 8);
    }

    #[test]
    fn the_payload_exposes_the_exact_solution_and_its_value() {
        let dir = tempfile::tempdir().unwrap();
        let solution: Vec<f64> = (0..13).map(|i| i as f64 * 0.1).collect();

        let run_dir =
            write_run_results(dir.path(), &metadata(&solution), "GRAHF", 42, 1000, 7.25).unwrap();

        let payload: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(run_dir.join("results.json")).unwrap())
                .unwrap();

        assert_eq!(payload["best_value"], 7.25);
        assert_eq!(payload["solution_dim"], 13);
        assert_eq!(payload["natural_dimension"], 60);
        assert_eq!(payload["n_evals"], 1000);
        assert_eq!(payload["x"].as_array().unwrap().len(), 13);
        assert_eq!(
            payload["export_schema_version"],
            OUTPUT_EXPORT_SCHEMA_VERSION
        );
        assert_eq!(payload["mode"], OUTPUT_ADAPTIVE_MODE);
        assert_eq!(payload["output_best_scope"], OUTPUT_BEST_SCOPE);
        assert_eq!(payload["output_transform_method"], "none");
    }

    #[test]
    fn the_initial_best_value_csv_holds_the_reported_value() {
        let dir = tempfile::tempdir().unwrap();
        let solution = vec![0.0; 36];

        let run_dir =
            write_run_results(dir.path(), &metadata(&solution), "GRAHF", 5, 42, 3.5).unwrap();

        let csv = fs::read_to_string(run_dir.join("best_value.csv")).unwrap();
        assert_eq!(csv.lines().next(), Some("nfes,best_value"));
        assert_eq!(csv.lines().nth(1), Some("42,3.5"));
    }

    #[test]
    fn writing_the_same_run_twice_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let solution = vec![0.0; 13];

        let first =
            write_run_results(dir.path(), &metadata(&solution), "GRAHF", 42, 10, 1.0).unwrap();
        let second =
            write_run_results(dir.path(), &metadata(&solution), "GRAHF", 42, 10, 1.0).unwrap();

        assert_eq!(first, second);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn writing_below_a_missing_root_reports_the_path() {
        let solution = vec![0.0; 13];

        let error = write_run_results(
            Path::new("/proc/definitely/not/writable"),
            &metadata(&solution),
            "GRAHF",
            1,
            1,
            1.0,
        )
        .unwrap_err();

        assert!(error.to_string().contains("failed to create"), "{error}");
    }
}
