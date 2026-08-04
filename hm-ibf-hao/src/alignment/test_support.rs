//! Instance fixtures shared by the unit tests of this module.

use ndarray::Array2;

use super::problem::HorizontalAlignment;

/// Allowed island working dimensions of the fixture instances.
///
/// Kept small and unrelated to the shipped `params_training.conf` so that a change to the
/// benchmark's own dimension set cannot silently alter what the unit tests exercise.
pub(crate) const FIXTURE_DIMENSIONS: [u32; 3] = [3, 5, 13];

/// Builds a straight, flat instance whose backbone runs along the first heightmap axis.
///
/// The terrain is level and the backbone is a straight line of length 40, so a zero-offset
/// solution costs exactly 40 and every deviation from that is attributable to the offsets.
///
/// # Returns
///
/// The fixture instance.
pub(crate) fn flat_instance() -> HorizontalAlignment {
    let points: Vec<[f64; 2]> = (0..=40).map(|i| [i as f64, 20.0]).collect();
    let distances: Vec<f64> = (0..=40).map(|i| i as f64).collect();

    HorizontalAlignment {
        name: "flat".to_string(),
        heightmap: Array2::zeros((41, 41)),
        backbone_normals: vec![[0.0, 1.0]; points.len()],
        backbone_offset_bounds: vec![(-10.0, 10.0); points.len()],
        backbone_points: points,
        backbone_cumulative_distances: distances,
        backbone_total_length: 40.0,
        natural_dimension: 3,
        simplified_path: vec![[0.0, 20.0], [40.0, 20.0]],
        tunnel_factor: 5.0,
        gradient_factor: 2.0,
        curvature_radius: 100.0,
        gradient_change_limit: 0.08,
        height_limit: 800.0,
        tau: 0.4,
        dimensions_allowed: FIXTURE_DIMENSIONS.to_vec(),
    }
}
