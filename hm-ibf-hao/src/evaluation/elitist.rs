//! Loading of a trained GRAHF elitist: its island graph and its IRACE-tuned parameters.

use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

use eyre::WrapErr;
use grahf::{graph::DiIntGraph, problems::algorithm_design::builder::BuilderGraph};
use mahf::{
    params::{Param, Params},
    ExecResult,
};
use serde::{Deserialize, Serialize};

use crate::{
    islands::island_builders,
    migrations::migration_builders,
    problems::{DimensionAwareDomain, RealValuedProblem},
};

/// A JSON-representable IRACE parameter value.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// An integer parameter.
    Integer(u32),
    /// A real-valued parameter.
    Float(f64),
    /// A categorical parameter.
    String(String),
    /// A nested parameter namespace.
    Nested(SerializableParams),
}

impl From<Value> for Param {
    fn from(value: Value) -> Self {
        match value {
            Value::Integer(i) => Param::new(i),
            Value::Float(f) => Param::new(f),
            Value::String(s) => Param::new(s),
            Value::Nested(map) => Param::new(Params::from(map)),
        }
    }
}

/// A JSON-representable parameter namespace.
#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SerializableParams {
    /// The parameters of this namespace, keyed by name.
    params: HashMap<String, Value>,
}

impl From<SerializableParams> for Params {
    fn from(value: SerializableParams) -> Self {
        let map: HashMap<_, _> = value
            .params
            .into_iter()
            .map(|(name, value)| (name, Param::from(value)))
            .collect();
        Params::from(map)
    }
}

/// Reads a stored island graph and maps its integer weights back to builders.
///
/// # Arguments
///
/// * `path` - The `<elitist>.json` file written by the training stage.
/// * `dimensions_allowed` - Allowed island working dimensions; see
///   [`crate::config::TrainingParams::dimensions_allowed`].
/// * `max_iterations` - Upper bound IRACE may assign to an island's iteration count.
/// * `max_population_size` - Upper bound IRACE may assign to an island's population size.
///
/// # Returns
///
/// The builder graph described by the stored weights.
///
/// # Errors
///
/// Returns an error if the file is missing, malformed, or contains a weight that does not
/// map to a known island or migration builder.
pub fn read_builder_graph<P: RealValuedProblem + DimensionAwareDomain>(
    path: &Path,
    dimensions_allowed: &[u32],
    max_iterations: u32,
    max_population_size: u32,
) -> ExecResult<BuilderGraph<P>> {
    let file = File::open(path).wrap_err_with(|| format!("failed to open {}", path.display()))?;
    let graph: DiIntGraph = serde_json::from_reader(BufReader::new(file))
        .wrap_err_with(|| format!("failed to parse {}", path.display()))?;

    let island_builders = island_builders(dimensions_allowed, max_iterations, max_population_size);
    let migration_builders = migration_builders();

    graph.try_map(
        |_, &weight| {
            island_builders
                .get(weight as usize)
                .cloned()
                .ok_or_else(|| unknown_weight("island", weight, island_builders.len()))
        },
        |_, &weight| {
            migration_builders
                .get(weight as usize)
                .cloned()
                .ok_or_else(|| unknown_weight("migration", weight, migration_builders.len()))
        },
    )
}

/// Builds the error reported for a weight that has no matching builder.
fn unknown_weight(kind: &str, weight: u32, available: usize) -> eyre::Report {
    eyre::eyre!("stored graph uses unknown {kind} type {weight}; only 0..{available} exist")
}

/// Reads the IRACE parameters stored alongside a trained elitist.
///
/// The training stage writes the parameters as a debug string, so the payload may be a JSON
/// string containing JSON, an object with a `_debug` field, or a plain object.
///
/// # Arguments
///
/// * `path` - The `<elitist>.params` file written by the training stage.
///
/// # Returns
///
/// The parsed parameters.
///
/// # Errors
///
/// Returns an error if the file is missing or does not contain a parameter namespace.
pub fn read_params(path: &Path) -> ExecResult<Params> {
    let file = File::open(path).wrap_err_with(|| format!("failed to open {}", path.display()))?;
    let content: serde_json::Value = serde_json::from_reader(BufReader::new(file))
        .wrap_err_with(|| format!("failed to parse {}", path.display()))?;

    let json_value = if let Some(s) = content.as_str() {
        serde_json::from_str::<serde_json::Value>(s)
            .wrap_err_with(|| format!("{} holds a string that is not JSON", path.display()))?
    } else if let Some(debug_str) = content.get("_debug").and_then(|v| v.as_str()) {
        serde_json::from_str::<serde_json::Value>(debug_str)
            .wrap_err_with(|| format!("{} holds a _debug field that is not JSON", path.display()))?
    } else {
        content
    };

    let serialized_params: SerializableParams = serde_json::from_value(json_value)
        .wrap_err_with(|| format!("{} is not a parameter namespace", path.display()))?;
    Ok(serialized_params.into())
}

/// Loads a trained elitist's builder graph and tuned parameters from `run_dir`.
///
/// # Arguments
///
/// * `run_dir` - Directory holding `<elitist>.json` and `<elitist>.params`.
/// * `elitist` - Base name of the artefacts, e.g. `elitist_0`.
/// * `dimensions_allowed` - Allowed island working dimensions; see
///   [`crate::config::TrainingParams::dimensions_allowed`].
/// * `max_iterations` - Upper bound IRACE may assign to an island's iteration count.
/// * `max_population_size` - Upper bound IRACE may assign to an island's population size.
///
/// # Returns
///
/// The builder graph and its tuned parameters.
///
/// # Errors
///
/// Returns an error if either artefact is missing or malformed.
pub fn load_elitist<P: RealValuedProblem + DimensionAwareDomain>(
    run_dir: &Path,
    elitist: &str,
    dimensions_allowed: &[u32],
    max_iterations: u32,
    max_population_size: u32,
) -> ExecResult<(BuilderGraph<P>, Params)> {
    let builder_graph = read_builder_graph(
        &run_dir.join(format!("{elitist}.json")),
        dimensions_allowed,
        max_iterations,
        max_population_size,
    )?;
    let params = read_params(&run_dir.join(format!("{elitist}.params")))?;
    Ok((builder_graph, params))
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;
    use crate::alignment::HorizontalAlignment;

    /// Allowed island dimensions used by the tests in this module.
    const TEST_DIMENSIONS: [u32; 3] = [53, 60, 68];

    /// Writes `content` to `<dir>/<name>` and returns the path.
    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, content).unwrap();
        path
    }

    /// Returns the error of a builder-graph read, panicking if it unexpectedly succeeded.
    ///
    /// `BuilderGraph` is not `Debug`, so `Result::unwrap_err` cannot be used here.
    fn graph_error(result: ExecResult<BuilderGraph<HorizontalAlignment>>) -> eyre::Report {
        match result {
            Ok(_) => panic!("expected the builder graph to be rejected"),
            Err(error) => error,
        }
    }

    #[test]
    fn a_missing_graph_file_reports_its_path() {
        let error = graph_error(read_builder_graph::<HorizontalAlignment>(
            Path::new("missing/elitist_0.json"),
            &TEST_DIMENSIONS,
            10,
            10,
        ));

        assert!(
            error.to_string().contains("missing/elitist_0.json"),
            "{error}"
        );
    }

    #[test]
    fn a_graph_with_an_unknown_island_weight_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        // Node weight 99 is far beyond the available island builders.
        let graph =
            r#"{"graph":{"nodes":[99],"node_holes":[],"edge_property":"directed","edges":[]}}"#;
        let path = write(dir.path(), "elitist_0.json", graph);

        let error = graph_error(read_builder_graph::<HorizontalAlignment>(
            &path,
            &TEST_DIMENSIONS,
            10,
            10,
        ));

        assert!(
            error.to_string().contains("unknown island type 99"),
            "{error}"
        );
    }

    #[test]
    fn a_valid_single_node_graph_is_mapped_to_builders() {
        let dir = tempfile::tempdir().unwrap();
        let graph =
            r#"{"graph":{"nodes":[0],"node_holes":[],"edge_property":"directed","edges":[]}}"#;
        let path = write(dir.path(), "elitist_0.json", graph);

        let builder_graph =
            read_builder_graph::<HorizontalAlignment>(&path, &TEST_DIMENSIONS, 10, 10).unwrap();

        assert_eq!(builder_graph.node_count(), 1);
        assert_eq!(builder_graph.edge_count(), 0);
    }

    #[test]
    fn a_malformed_graph_file_reports_its_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "elitist_0.json", "not json at all");

        let error = graph_error(read_builder_graph::<HorizontalAlignment>(
            &path,
            &TEST_DIMENSIONS,
            10,
            10,
        ));

        assert!(error.to_string().contains("failed to parse"), "{error}");
    }

    #[test]
    fn params_are_read_from_a_plain_object() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(
            dir.path(),
            "p.params",
            r#"{"island":{"0":{"dimension":53}}}"#,
        );

        let params = read_params(&path).unwrap();

        assert!(params.try_get::<Params>("island").is_ok());
    }

    #[test]
    fn params_are_read_from_a_json_encoded_string() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "p.params", r#""{\"a\":1}""#);

        assert!(read_params(&path).is_ok());
    }

    #[test]
    fn params_are_read_from_a_debug_wrapper() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "p.params", r#"{"_debug":"{\"a\":1}"}"#);

        assert!(read_params(&path).is_ok());
    }

    #[test]
    fn a_params_file_that_is_not_a_namespace_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "p.params", "[1, 2, 3]");

        let error = read_params(&path).unwrap_err();

        assert!(
            error.to_string().contains("not a parameter namespace"),
            "{error}"
        );
    }

    #[test]
    fn load_elitist_reports_the_missing_artefact() {
        let dir = tempfile::tempdir().unwrap();

        let error = match load_elitist::<HorizontalAlignment>(
            dir.path(),
            "elitist_0",
            &TEST_DIMENSIONS,
            10,
            10,
        ) {
            Ok(_) => panic!("expected the missing artefact to be reported"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("elitist_0.json"), "{error}");
    }
}
