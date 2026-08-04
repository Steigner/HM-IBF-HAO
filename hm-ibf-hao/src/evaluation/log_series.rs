//! Extraction of per-iteration progress series from a MAHF run log.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use eyre::{bail, WrapErr};
use mahf::ExecResult;
use serde_json::Value as JsonValue;

/// One `(evaluations, value)` sample of a progress series.
pub type Sample = (u64, f64);

/// The progress series extracted from one run log.
#[derive(Debug, Default, PartialEq)]
pub struct ProgressSeries {
    /// Best objective value against the evaluation count.
    pub best: Vec<Sample>,
    /// Mean island objective value against the evaluation count.
    pub average: Vec<Sample>,
}

/// Parses the best and mean-island progress series from a MAHF log.
///
/// MAHF stores log entries as objects keyed by the field index and lists the field names in
/// a parallel `names` array, so the indices are resolved by name rather than by position.
///
/// # Arguments
///
/// * `log` - The parsed contents of a `run.log` JSON file.
///
/// # Returns
///
/// The best and average series, both ordered as they appear in the log.
///
/// # Errors
///
/// Returns an error if the log has no `names` array or does not contain the required
/// `Evaluations` and `BestObjectiveValue` fields. Failing here is deliberate: silently
/// guessing the field positions would produce plausible but wrong progress curves.
pub fn parse_log_series(log: &JsonValue) -> ExecResult<ProgressSeries> {
    let Some(names) = log.get("names").and_then(JsonValue::as_array) else {
        bail!("run log has no `names` array, so its fields cannot be resolved");
    };

    let position = |predicate: &dyn Fn(&str) -> bool| {
        names
            .iter()
            .position(|name| name.as_str().is_some_and(predicate))
    };

    let idx_best = position(&|name| name.contains("BestObjectiveValue"))
        .ok_or_else(|| eyre::eyre!("run log has no `BestObjectiveValue` field"))?;
    let idx_evals = position(&|name| name.contains("Evaluations"))
        .ok_or_else(|| eyre::eyre!("run log has no `Evaluations` field"))?;
    // The per-island values are optional: only the evaluation binary registers them.
    let idx_island = position(&|name| name.contains("ObjectiveValues") && !name.contains("Best"));

    let mut series = ProgressSeries::default();
    let Some(entries) = log.get("entries").and_then(JsonValue::as_array) else {
        return Ok(series);
    };

    for entry in entries {
        let evaluations = entry
            .get(idx_evals.to_string())
            .and_then(JsonValue::as_u64)
            .unwrap_or(0);

        if let Some(best) = entry.get(idx_best.to_string()).and_then(JsonValue::as_f64) {
            series.best.push((evaluations, best));
        }

        let island_values = idx_island
            .and_then(|idx| entry.get(idx.to_string()))
            .and_then(JsonValue::as_array);
        if let Some(values) = island_values {
            let values: Vec<f64> = values.iter().filter_map(JsonValue::as_f64).collect();
            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                series.average.push((evaluations, mean));
            }
        }
    }

    Ok(series)
}

/// Writes the progress series of one run as `best_value.csv` and `avg_value.csv`.
///
/// # Arguments
///
/// * `dir` - The run's result directory, which must already exist.
/// * `series` - The series to write.
///
/// # Errors
///
/// Returns an error if either file cannot be created or written.
pub fn write_progress_csvs(dir: &Path, series: &ProgressSeries) -> ExecResult<()> {
    write_series(&dir.join("best_value.csv"), "best_value", &series.best)?;
    write_series(&dir.join("avg_value.csv"), "avg_value", &series.average)?;
    Ok(())
}

/// Writes a single `nfes,<column>` CSV file.
fn write_series(path: &Path, column: &str, samples: &[Sample]) -> ExecResult<()> {
    let file =
        File::create(path).wrap_err_with(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "nfes,{column}")?;
    for (evaluations, value) in samples {
        writeln!(writer, "{evaluations},{value}")?;
    }
    writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Builds a log with the field layout the evaluation stage registers.
    fn log() -> JsonValue {
        json!({
            "names": ["Iterations", "BestObjectiveValue", "Evaluations", "ObjectiveValues"],
            "entries": [
                {"0": 1, "1": 5.0, "2": 100, "3": [4.0, 6.0]},
                {"0": 2, "1": 3.0, "2": 200, "3": [2.0, 4.0]},
            ]
        })
    }

    #[test]
    fn series_are_resolved_by_field_name() {
        let series = parse_log_series(&log()).unwrap();

        assert_eq!(series.best, vec![(100, 5.0), (200, 3.0)]);
        assert_eq!(series.average, vec![(100, 5.0), (200, 3.0)]);
    }

    #[test]
    fn reordered_fields_are_still_resolved_correctly() {
        let reordered = json!({
            "names": ["Iterations", "Evaluations", "BestObjectiveValue"],
            "entries": [{"0": 1, "1": 700, "2": 1.5}]
        });

        let series = parse_log_series(&reordered).unwrap();

        assert_eq!(series.best, vec![(700, 1.5)]);
        assert!(series.average.is_empty());
    }

    #[test]
    fn a_log_without_names_is_rejected() {
        // Regression: the previous implementation fell back to hardcoded field positions,
        // which silently produced wrong curves whenever the log layout changed.
        let error = parse_log_series(&json!({"entries": []})).unwrap_err();

        assert!(error.to_string().contains("`names`"), "{error}");
    }

    #[test]
    fn a_log_without_the_required_fields_is_rejected() {
        let error = parse_log_series(&json!({"names": ["Iterations"], "entries": []})).unwrap_err();

        assert!(error.to_string().contains("BestObjectiveValue"), "{error}");
    }

    #[test]
    fn a_log_without_entries_yields_empty_series() {
        let empty = json!({"names": ["Iterations", "BestObjectiveValue", "Evaluations"]});

        assert_eq!(parse_log_series(&empty).unwrap(), ProgressSeries::default());
    }

    #[test]
    fn empty_island_value_arrays_are_skipped() {
        let sparse = json!({
            "names": ["Iterations", "BestObjectiveValue", "Evaluations", "ObjectiveValues"],
            "entries": [{"0": 1, "1": 5.0, "2": 100, "3": []}]
        });

        let series = parse_log_series(&sparse).unwrap();

        assert_eq!(series.best.len(), 1);
        assert!(series.average.is_empty());
    }

    #[test]
    fn csvs_contain_a_header_and_one_row_per_sample() {
        let dir = tempfile::tempdir().unwrap();
        let series = parse_log_series(&log()).unwrap();

        write_progress_csvs(dir.path(), &series).unwrap();

        let best = std::fs::read_to_string(dir.path().join("best_value.csv")).unwrap();
        assert_eq!(best.lines().next(), Some("nfes,best_value"));
        assert_eq!(best.lines().count(), 3);

        let average = std::fs::read_to_string(dir.path().join("avg_value.csv")).unwrap();
        assert_eq!(average.lines().next(), Some("nfes,avg_value"));
        assert_eq!(average.lines().count(), 3);
    }

    #[test]
    fn writing_into_a_missing_directory_reports_the_path() {
        let error =
            write_progress_csvs(Path::new("no/such/dir"), &ProgressSeries::default()).unwrap_err();

        assert!(error.to_string().contains("best_value.csv"), "{error}");
    }
}
