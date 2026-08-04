//! On-disk instance format and instance loading.
//!
//! Preprocessing writes one pair of files per instance: `<name>_config.json` holds the
//! backbone and the cost parameters, `<name>_heightmap.npy` the terrain it refers to.

use std::{
    fs,
    path::{Path, PathBuf},
};

use eyre::{ensure, WrapErr};
use mahf::ExecResult;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::problem::HorizontalAlignment;

/// Suffix of the config file of an instance.
const CONFIG_SUFFIX: &str = "_config";

/// Default clothoid asymmetry for configs written before `tau` became a parameter.
const DEFAULT_TAU: f64 = 0.4;

/// The backbone section of an instance config.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackboneConfig {
    /// Equidistant backbone points.
    pub points: Vec<Vec<f64>>,
    /// Cumulative arc lengths, aligned with `points`.
    pub cumulative_distances: Vec<f64>,
    /// Total arc length of the backbone.
    pub total_length: f64,
    /// Unit normals, aligned with `points`.
    pub normals: Vec<Vec<f64>>,
    /// `[min, max]` offset bounds, aligned with `points`.
    pub offset_bounds: Vec<Vec<f64>>,
}

/// The config file of a single instance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlignmentConfig {
    /// Instance name; matches the file name stem.
    pub name: String,
    /// Shape of the heightmap the backbone refers to.
    pub heightmap_shape: Vec<usize>,
    /// Route start in heightmap coordinates.
    pub start: Vec<i32>,
    /// Route goal in heightmap coordinates.
    pub goal: Vec<i32>,
    /// The least-cost route the backbone was resampled from.
    pub path_astar: Vec<Vec<i32>>,
    /// The simplified route the natural dimension was derived from.
    pub path_simplified: Vec<Vec<i32>>,
    /// The backbone.
    pub backbone: BackboneConfig,
    /// Simplification tolerance used during preprocessing.
    pub epsilon: f64,
    /// Multiplier that scaled the offset bounds during preprocessing.
    pub cutting_plane_factor: f64,
    /// Cost multiplier for tunnelled segments.
    pub tunnel_factor: f64,
    /// Cost multiplier for steep segments.
    pub gradient_factor: f64,
    /// Smallest allowed radius of curvature.
    pub curvature_radius: f64,
    /// Largest unpenalized absolute gradient.
    pub gradient_change_limit: f64,
    /// Elevation above which the tunnel penalty applies.
    pub height_limit: f64,
    /// Clothoid asymmetry parameter.
    #[serde(default = "default_tau")]
    pub tau: f64,
}

/// Returns the clothoid asymmetry assumed for configs that omit it.
fn default_tau() -> f64 {
    DEFAULT_TAU
}

impl HorizontalAlignment {
    /// Loads every instance in a directory, sorted by name.
    ///
    /// A directory entry is an instance when it is named `<name>_config.json` and a matching
    /// `<name>_heightmap.npy` exists next to it, which is what keeps `summary.json` and other
    /// bookkeeping files out of the benchmark.
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory holding the instance files.
    /// * `dimensions_allowed` - Allowed island working dimensions; see
    ///   [`crate::config::TrainingParams::dimensions_allowed`].
    ///
    /// # Returns
    ///
    /// The loaded instances, in a deterministic order.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read, an instance is malformed, or the
    /// directory contains no instance at all.
    pub fn load_instances(
        dir: impl AsRef<Path>,
        dimensions_allowed: &[u32],
    ) -> ExecResult<Vec<Self>> {
        let dir = dir.as_ref();
        let mut config_paths: Vec<PathBuf> = fs::read_dir(dir)
            .wrap_err_with(|| format!("failed to read {}", dir.display()))?
            .collect::<Result<Vec<_>, _>>()
            .wrap_err_with(|| format!("failed to list {}", dir.display()))?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| instance_name(path).is_some())
            .collect();
        config_paths.sort();

        let instances: Vec<Self> = config_paths
            .iter()
            .filter_map(|config_path| {
                let name = instance_name(config_path)?;
                let heightmap_path = dir.join(format!("{name}_heightmap.npy"));
                heightmap_path
                    .is_file()
                    .then(|| Self::from_files(config_path, &heightmap_path, dimensions_allowed))
            })
            .collect::<ExecResult<_>>()?;

        ensure!(
            !instances.is_empty(),
            "no instances found in {}; expected <name>_config.json plus <name>_heightmap.npy",
            dir.display()
        );

        Ok(instances)
    }

    /// Loads a single instance from its config and heightmap files.
    ///
    /// # Arguments
    ///
    /// * `config_path` - The `<name>_config.json` file.
    /// * `heightmap_path` - The `<name>_heightmap.npy` file.
    /// * `dimensions_allowed` - Allowed island working dimensions; see
    ///   [`crate::config::TrainingParams::dimensions_allowed`].
    ///
    /// # Returns
    ///
    /// The loaded instance.
    ///
    /// # Errors
    ///
    /// Returns an error if either file is missing or malformed, or if the backbone's arrays
    /// disagree in length.
    pub fn from_files(
        config_path: &Path,
        heightmap_path: &Path,
        dimensions_allowed: &[u32],
    ) -> ExecResult<Self> {
        let config = fs::read_to_string(config_path)
            .wrap_err_with(|| format!("failed to read {}", config_path.display()))?;
        let config: AlignmentConfig = serde_json::from_str(&config)
            .wrap_err_with(|| format!("failed to parse {}", config_path.display()))?;

        let heightmap = read_heightmap(heightmap_path)?;
        Self::from_config(config, heightmap, dimensions_allowed)
            .wrap_err_with(|| format!("invalid instance {}", config_path.display()))
    }

    /// Builds an instance from an already parsed config and heightmap.
    ///
    /// # Arguments
    ///
    /// * `config` - The parsed config.
    /// * `heightmap` - The terrain the backbone refers to.
    /// * `dimensions_allowed` - Allowed island working dimensions; see
    ///   [`crate::config::TrainingParams::dimensions_allowed`].
    ///
    /// # Returns
    ///
    /// The instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the backbone is too short to interpolate, its arrays disagree in
    /// length, or `dimensions_allowed` is empty.
    pub fn from_config(
        config: AlignmentConfig,
        heightmap: Array2<f64>,
        dimensions_allowed: &[u32],
    ) -> ExecResult<Self> {
        let points = to_pairs(&config.backbone.points);
        let normals = to_pairs(&config.backbone.normals);
        let bounds: Vec<(f64, f64)> = config
            .backbone
            .offset_bounds
            .iter()
            .map(|bound| (bound[0], bound[1]))
            .collect();

        ensure!(
            points.len() >= 2,
            "the backbone needs at least two points, got {}",
            points.len()
        );
        ensure!(
            normals.len() == points.len()
                && bounds.len() == points.len()
                && config.backbone.cumulative_distances.len() == points.len(),
            "the backbone arrays disagree: {} points, {} normals, {} bounds, {} distances",
            points.len(),
            normals.len(),
            bounds.len(),
            config.backbone.cumulative_distances.len()
        );
        ensure!(
            config.backbone.total_length > 0.0,
            "the backbone has no length"
        );
        ensure!(
            config.curvature_radius > 0.0,
            "the minimum radius of curvature must be positive"
        );
        // `dimension()` reads the last entry as the declared search space size, so an empty
        // list would make every island initialize at dimension zero.
        ensure!(
            !dimensions_allowed.is_empty(),
            "at least one island dimension must be allowed"
        );

        Ok(Self {
            name: config.name,
            heightmap,
            backbone_points: points,
            backbone_cumulative_distances: config.backbone.cumulative_distances,
            backbone_total_length: config.backbone.total_length,
            backbone_normals: normals,
            backbone_offset_bounds: bounds,
            // The two route endpoints are fixed boundary conditions and never optimized.
            natural_dimension: config.path_simplified.len().saturating_sub(2),
            simplified_path: config
                .path_simplified
                .iter()
                .map(|point| [point[0] as f64, point[1] as f64])
                .collect(),
            tunnel_factor: config.tunnel_factor,
            gradient_factor: config.gradient_factor,
            curvature_radius: config.curvature_radius,
            gradient_change_limit: config.gradient_change_limit,
            height_limit: config.height_limit,
            tau: config.tau,
            dimensions_allowed: dimensions_allowed.to_vec(),
        })
    }
}

/// Returns the instance name of a path, if it is an instance config file.
fn instance_name(path: &Path) -> Option<String> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
        return None;
    }

    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.strip_suffix(CONFIG_SUFFIX))
        .map(str::to_string)
}

/// Reads a heightmap from a NumPy `.npy` file.
///
/// Preprocessing stores the terrain as `float32`; it is widened here so the objective is
/// computed in the same precision as everything else.
///
/// # Arguments
///
/// * `path` - The `.npy` file.
///
/// # Returns
///
/// The heightmap.
///
/// # Errors
///
/// Returns an error if the file is missing or not a two-dimensional `float32` array.
fn read_heightmap(path: &Path) -> ExecResult<Array2<f64>> {
    let heightmap: Array2<f32> = ndarray_npy::read_npy(path)
        .map_err(|error| eyre::eyre!("{error}"))
        .wrap_err_with(|| format!("failed to read heightmap {}", path.display()))?;

    ensure!(
        heightmap.dim().0 >= 2 && heightmap.dim().1 >= 2,
        "heightmap {} is too small: {:?}",
        path.display(),
        heightmap.dim()
    );

    Ok(heightmap.mapv(f64::from))
}

/// Converts a list of coordinate pairs into fixed-size arrays.
fn to_pairs(values: &[Vec<f64>]) -> Vec<[f64; 2]> {
    values
        .iter()
        .filter(|value| value.len() >= 2)
        .map(|value| [value[0], value[1]])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allowed island dimensions used by the tests in this module.
    const TEST_DIMENSIONS: [u32; 3] = [3, 5, 13];

    /// Returns a minimal but valid config.
    fn config() -> AlignmentConfig {
        AlignmentConfig {
            name: "unit".to_string(),
            heightmap_shape: vec![4, 4],
            start: vec![0, 0],
            goal: vec![3, 3],
            path_astar: vec![vec![0, 0], vec![3, 3]],
            path_simplified: vec![vec![0, 0], vec![1, 1], vec![2, 2], vec![3, 3]],
            backbone: BackboneConfig {
                points: vec![vec![0.0, 0.0], vec![3.0, 3.0]],
                cumulative_distances: vec![0.0, 4.242_640_687],
                total_length: 4.242_640_687,
                normals: vec![vec![0.0, 1.0], vec![0.0, 1.0]],
                offset_bounds: vec![vec![-1.0, 1.0], vec![-1.0, 1.0]],
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

    #[test]
    fn a_valid_config_builds_an_instance() {
        let instance =
            HorizontalAlignment::from_config(config(), Array2::zeros((4, 4)), &TEST_DIMENSIONS)
                .unwrap();

        assert_eq!(instance.name, "unit");
        assert_eq!(instance.natural_dimension, 2, "four points minus both ends");
        assert_eq!(instance.backbone_points.len(), 2);
    }

    #[test]
    fn a_backbone_with_mismatched_arrays_is_rejected() {
        let mut config = config();
        config.backbone.normals.pop();

        let error =
            HorizontalAlignment::from_config(config, Array2::zeros((4, 4)), &TEST_DIMENSIONS)
                .unwrap_err();

        assert!(error.to_string().contains("disagree"), "{error}");
    }

    #[test]
    fn a_backbone_of_one_point_is_rejected() {
        let mut config = config();
        config.backbone.points.pop();
        config.backbone.normals.pop();
        config.backbone.offset_bounds.pop();
        config.backbone.cumulative_distances.pop();

        let error =
            HorizontalAlignment::from_config(config, Array2::zeros((4, 4)), &TEST_DIMENSIONS)
                .unwrap_err();

        assert!(error.to_string().contains("at least two points"), "{error}");
    }

    #[test]
    fn a_zero_length_backbone_is_rejected() {
        let mut config = config();
        config.backbone.total_length = 0.0;

        let error =
            HorizontalAlignment::from_config(config, Array2::zeros((4, 4)), &TEST_DIMENSIONS)
                .unwrap_err();

        assert!(error.to_string().contains("no length"), "{error}");
    }

    #[test]
    fn a_non_positive_curvature_radius_is_rejected() {
        let mut config = config();
        config.curvature_radius = 0.0;

        let error =
            HorizontalAlignment::from_config(config, Array2::zeros((4, 4)), &TEST_DIMENSIONS)
                .unwrap_err();

        assert!(error.to_string().contains("radius of curvature"), "{error}");
    }

    #[test]
    fn tau_defaults_when_the_config_omits_it() {
        let mut value = serde_json::to_value(config()).unwrap();
        value.as_object_mut().unwrap().remove("tau");

        let parsed: AlignmentConfig = serde_json::from_value(value).unwrap();

        assert_eq!(parsed.tau, DEFAULT_TAU);
    }

    #[test]
    fn only_config_files_are_recognised_as_instances() {
        assert_eq!(
            instance_name(Path::new("/x/alps_config.json")).as_deref(),
            Some("alps")
        );
        assert!(instance_name(Path::new("/x/summary.json")).is_none());
        assert!(instance_name(Path::new("/x/alps_heightmap.npy")).is_none());
        assert!(instance_name(Path::new("/x/alps_config.txt")).is_none());
    }

    #[test]
    fn an_empty_directory_holds_no_instances() {
        let dir = tempfile::tempdir().unwrap();

        let error = HorizontalAlignment::load_instances(dir.path(), &TEST_DIMENSIONS).unwrap_err();

        assert!(error.to_string().contains("no instances found"), "{error}");
    }

    #[test]
    fn a_missing_directory_reports_its_path() {
        let error =
            HorizontalAlignment::load_instances("no/such/dir", &TEST_DIMENSIONS).unwrap_err();

        assert!(error.to_string().contains("no/such/dir"), "{error}");
    }

    #[test]
    fn a_config_without_its_heightmap_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("alps_config.json"),
            serde_json::to_string(&config()).unwrap(),
        )
        .unwrap();

        let error = HorizontalAlignment::load_instances(dir.path(), &TEST_DIMENSIONS).unwrap_err();

        assert!(error.to_string().contains("no instances found"), "{error}");
    }

    #[test]
    fn a_missing_heightmap_file_reports_its_path() {
        let error = read_heightmap(Path::new("no/such/map.npy")).unwrap_err();

        assert!(error.to_string().contains("no/such/map.npy"), "{error}");
    }

    #[test]
    fn short_coordinate_pairs_are_dropped() {
        let pairs = to_pairs(&[vec![1.0, 2.0], vec![3.0], vec![4.0, 5.0, 6.0]]);

        assert_eq!(pairs, vec![[1.0, 2.0], [4.0, 5.0]]);
    }
}
