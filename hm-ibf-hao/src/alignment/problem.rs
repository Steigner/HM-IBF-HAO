//! The problem definition: instance geometry, terrain lookups and the objective function.

use std::ops::Range;

use mahf::{
    problems::{KnownOptimumProblem, LimitedVectorProblem, VectorProblem},
    Problem, SingleObjective,
};
use ndarray::Array2;

use super::{samples::BackboneSamples, CLOTHOID_POINT_STEP, CURVATURE_PENALTY};
use crate::{
    clothoid::{check_curvature_constraint_numerical, interpolate_path_with_tau},
    problems::DimensionAwareDomain,
};

/// Distances below this are treated as zero when interpolating along the backbone.
const EPSILON: f64 = 1e-10;

/// A single horizontal alignment instance.
///
/// The backbone and the terrain come from preprocessing and never change; only the offset
/// vector handed to [`HorizontalAlignment::evaluate_solution`] varies during a run.
#[derive(Clone, Debug)]
pub struct HorizontalAlignment {
    /// Instance name, used for logging and output filenames.
    pub name: String,
    /// Downsampled elevation heightmap, indexed `[row, column]`.
    pub heightmap: Array2<f64>,
    /// Equidistant backbone control points.
    pub backbone_points: Vec<[f64; 2]>,
    /// Cumulative arc lengths along the backbone, aligned with `backbone_points`.
    pub backbone_cumulative_distances: Vec<f64>,
    /// Total arc length of the backbone.
    pub backbone_total_length: f64,
    /// Unit normals perpendicular to the backbone at each point.
    pub backbone_normals: Vec<[f64; 2]>,
    /// Inclusive `(min, max)` offset bounds at each backbone point, clipped to the terrain.
    pub backbone_offset_bounds: Vec<(f64, f64)>,
    /// Number of interior inflection points of the simplified route.
    ///
    /// A proxy for the instance's geometric complexity, used when choosing the working
    /// dimensions the islands may run at.
    pub natural_dimension: usize,
    /// The simplified route the `natural_dimension` was derived from.
    pub simplified_path: Vec<[f64; 2]>,
    /// Cost multiplier for segments above `height_limit`, which have to be tunnelled.
    pub tunnel_factor: f64,
    /// Cost multiplier for segments steeper than `gradient_change_limit`.
    pub gradient_factor: f64,
    /// Smallest radius of curvature the route may have.
    pub curvature_radius: f64,
    /// Largest absolute gradient a segment may have before it is penalized.
    pub gradient_change_limit: f64,
    /// Elevation above which the tunnel penalty applies.
    pub height_limit: f64,
    /// Clothoid asymmetry parameter in `[0, 1]`.
    pub tau: f64,
    /// Allowed island working dimensions, strictly increasing.
    ///
    /// See [`crate::config::TrainingParams::dimensions_allowed`]. Only the largest entry is
    /// read here, to size the declared search space.
    pub(crate) dimensions_allowed: Vec<u32>,
}

impl HorizontalAlignment {
    /// Returns the terrain elevation at a point, clamped to the heightmap.
    ///
    /// # Arguments
    ///
    /// * `x` - Row coordinate.
    /// * `y` - Column coordinate.
    ///
    /// # Returns
    ///
    /// The elevation of the nearest heightmap cell.
    #[inline]
    pub fn height_at(&self, x: f64, y: f64) -> f64 {
        let (rows, columns) = self.heightmap.dim();
        let x = x.clamp(0.0, (rows - 1) as f64);
        let y = y.clamp(0.0, (columns - 1) as f64);
        self.heightmap[[x as usize, y as usize]]
    }

    /// Returns the terrain gradient between two points.
    ///
    /// # Arguments
    ///
    /// * `from` - Start of the segment.
    /// * `to` - End of the segment.
    ///
    /// # Returns
    ///
    /// Rise over run, or zero for a segment of vanishing horizontal length.
    #[inline]
    pub fn gradient_between(&self, from: [f64; 2], to: [f64; 2]) -> f64 {
        let run = ((to[0] - from[0]).powi(2) + (to[1] - from[1]).powi(2)).sqrt();
        if run < EPSILON {
            return 0.0;
        }
        (self.height_at(to[0], to[1]) - self.height_at(from[0], from[1])) / run
    }

    /// Clamps a point to the heightmap and snaps it to the nearest cell centre.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to clamp.
    ///
    /// # Returns
    ///
    /// The clamped point.
    #[inline]
    pub fn clip_point(&self, point: [f64; 2]) -> [f64; 2] {
        let (rows, columns) = self.heightmap.dim();
        [
            point[0].clamp(0.0, (rows - 1) as f64).trunc(),
            point[1].clamp(0.0, (columns - 1) as f64).trunc(),
        ]
    }

    /// Samples `dimension` equidistant control positions along the backbone.
    ///
    /// The positions sit at `s_i = i·L/(D+1)` for `i = 1..=D`, so the backbone's own start
    /// and end are excluded: they are fixed boundary conditions with zero offset.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Number of control points to place.
    ///
    /// # Returns
    ///
    /// The positions, their unit normals and their offset bounds, all of length `dimension`.
    pub fn sample_backbone(&self, dimension: usize) -> BackboneSamples {
        let mut samples = BackboneSamples::with_capacity(dimension);

        for i in 1..=dimension {
            let distance = i as f64 * self.backbone_total_length / (dimension as f64 + 1.0);
            let (position, normal, bounds) = self.interpolate_backbone_at(distance);
            samples.push(position, normal, bounds);
        }

        samples
    }

    /// Interpolates the backbone at a cumulative arc length.
    ///
    /// # Arguments
    ///
    /// * `distance` - Arc length from the start of the backbone, clamped to its total length.
    ///
    /// # Returns
    ///
    /// The interpolated position, unit normal and offset bounds.
    fn interpolate_backbone_at(&self, distance: f64) -> ([f64; 2], [f64; 2], (f64, f64)) {
        let distances = &self.backbone_cumulative_distances;
        let distance = distance.clamp(0.0, self.backbone_total_length);

        let index = match distances.binary_search_by(|d| d.total_cmp(&distance)) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        }
        .min(distances.len() - 2);

        let (start, end) = (distances[index], distances[index + 1]);
        let t = if (end - start).abs() < EPSILON {
            0.0
        } else {
            ((distance - start) / (end - start)).clamp(0.0, 1.0)
        };

        let (p0, p1) = (self.backbone_points[index], self.backbone_points[index + 1]);
        let position = [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])];

        let (n0, n1) = (
            self.backbone_normals[index],
            self.backbone_normals[index + 1],
        );
        let blended = [n0[0] + t * (n1[0] - n0[0]), n0[1] + t * (n1[1] - n0[1])];
        let magnitude = (blended[0] * blended[0] + blended[1] * blended[1]).sqrt();
        let normal = if magnitude > EPSILON {
            [blended[0] / magnitude, blended[1] / magnitude]
        } else {
            n0
        };

        let (b0, b1) = (
            self.backbone_offset_bounds[index],
            self.backbone_offset_bounds[index + 1],
        );
        let bounds = (b0.0 + t * (b1.0 - b0.0), b0.1 + t * (b1.1 - b0.1));

        (position, normal, bounds)
    }

    /// Builds the control polygon an offset vector describes.
    ///
    /// Each offset displaces its backbone position along the local normal; the backbone's
    /// endpoints bracket the result and are never displaced.
    ///
    /// # Arguments
    ///
    /// * `offsets` - The offset vector; its length selects the working dimension.
    ///
    /// # Returns
    ///
    /// The control polygon, of length `offsets.len() + 2`.
    pub fn control_polygon(&self, offsets: &[f64]) -> Vec<[f64; 2]> {
        let samples = self.sample_backbone(offsets.len());

        let mut polygon = Vec::with_capacity(offsets.len() + 2);
        polygon.push(self.backbone_points[0]);
        for (i, &offset) in offsets.iter().enumerate() {
            let offset = offset.clamp(samples.bounds[i].0, samples.bounds[i].1);
            polygon.push(self.clip_point([
                samples.positions[i][0] + offset * samples.normals[i][0],
                samples.positions[i][1] + offset * samples.normals[i][1],
            ]));
        }
        polygon.push(*self.backbone_points.last().unwrap());

        polygon
    }

    /// Scores an offset vector.
    ///
    /// # Arguments
    ///
    /// * `offsets` - The offset vector; its length selects the working dimension.
    ///
    /// # Returns
    ///
    /// The terrain-weighted route length plus [`CURVATURE_PENALTY`] per violating sample.
    pub fn evaluate_solution(&self, offsets: &[f64]) -> f64 {
        let route = interpolate_path_with_tau(
            &self.control_polygon(offsets),
            self.tau,
            CLOTHOID_POINT_STEP,
        );
        let violations = check_curvature_constraint_numerical(&route, 1.0 / self.curvature_radius);

        self.weighted_length(&route) + violations * CURVATURE_PENALTY
    }

    /// Returns the length of a route, weighted by the terrain it crosses.
    ///
    /// A segment above `height_limit` has to be tunnelled and a segment steeper than
    /// `gradient_change_limit` needs earthworks; a segment that is both pays both factors.
    ///
    /// # Arguments
    ///
    /// * `route` - The sampled route.
    ///
    /// # Returns
    ///
    /// The weighted length; zero for a route of fewer than two points.
    pub fn weighted_length(&self, route: &[[f64; 2]]) -> f64 {
        route
            .windows(2)
            .map(|segment| {
                let (from, to) = (segment[0], segment[1]);
                let length = ((to[0] - from[0]).powi(2) + (to[1] - from[1]).powi(2)).sqrt();

                let mut weight = 1.0;
                if self.height_at(to[0], to[1]) > self.height_limit {
                    weight *= self.tunnel_factor;
                }
                if self.gradient_between(from, to).abs() > self.gradient_change_limit {
                    weight *= self.gradient_factor;
                }
                length * weight
            })
            .sum()
    }
}

impl Problem for HorizontalAlignment {
    type Encoding = Vec<f64>;
    type Objective = SingleObjective;

    fn name(&self) -> &str {
        &self.name
    }
}

impl VectorProblem for HorizontalAlignment {
    type Element = f64;

    /// Returns the largest allowed island dimension.
    ///
    /// Islands run at their own IRACE-tuned dimension and read `solution.len()`; this value
    /// only sizes the declared search space.
    fn dimension(&self) -> usize {
        *self
            .dimensions_allowed
            .last()
            .expect("at least one island dimension must be allowed") as usize
    }
}

impl LimitedVectorProblem for HorizontalAlignment {
    fn domain(&self) -> Vec<Range<Self::Element>> {
        self.domain_for_dimension(self.dimension())
    }
}

impl KnownOptimumProblem for HorizontalAlignment {
    /// Returns zero: a route of zero length and no violations is the unreachable ideal.
    fn known_optimum(&self) -> SingleObjective {
        0.0.try_into().expect("zero is a valid objective value")
    }
}

impl DimensionAwareDomain for HorizontalAlignment {
    /// Returns bounds sampled at exactly `dim` equidistant backbone positions.
    ///
    /// Bounds are position-dependent, so slicing the maximum-dimension domain would hand an
    /// island the bounds of positions it does not use; the backbone is resampled instead.
    fn domain_for_dimension(&self, dim: usize) -> Vec<Range<f64>> {
        self.sample_backbone(dim)
            .bounds
            .iter()
            .map(|(low, high)| *low..*high)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{super::test_support::flat_instance, *};

    /// Allowed island dimensions the `flat_instance` fixture is built with.
    const TEST_DIMENSIONS: [u32; 3] = [3, 5, 13];

    #[test]
    fn sampling_places_control_points_strictly_inside_the_backbone() {
        let instance = flat_instance();

        let samples = instance.sample_backbone(4);

        assert_eq!(samples.len(), 4);
        assert!(!samples.is_empty());
        assert_eq!(samples.normals.len(), 4);
        assert_eq!(samples.bounds.len(), 4);
        // s_i = i * 40 / 5 = 8, 16, 24, 32.
        for (i, position) in samples.positions.iter().enumerate() {
            assert!(
                (position[0] - (8 * (i + 1)) as f64).abs() < 1e-9,
                "{:?}",
                samples.positions
            );
        }
    }

    #[test]
    fn different_dimensions_place_control_points_differently() {
        let instance = flat_instance();

        let three = instance.sample_backbone(3);
        let four = instance.sample_backbone(4);

        assert_ne!(
            three.positions[0], four.positions[0],
            "bounds are position dependent"
        );
        assert!(instance.sample_backbone(0).is_empty());
    }

    #[test]
    fn the_domain_is_resampled_per_dimension() {
        let instance = flat_instance();

        for dimension in [1, 5, 13, 79] {
            assert_eq!(instance.domain_for_dimension(dimension).len(), dimension);
        }
        assert_eq!(instance.domain().len(), instance.dimension());
        assert_eq!(
            instance.dimension(),
            *TEST_DIMENSIONS.last().unwrap() as usize
        );
    }

    #[test]
    fn heights_and_points_are_clamped_to_the_heightmap() {
        let instance = flat_instance();

        assert_eq!(instance.height_at(-5.0, -5.0), 0.0);
        assert_eq!(instance.height_at(1e9, 1e9), 0.0);
        assert_eq!(instance.clip_point([-3.0, 1e9]), [0.0, 40.0]);
    }

    #[test]
    fn a_flat_segment_has_no_gradient() {
        let instance = flat_instance();

        assert_eq!(instance.gradient_between([0.0, 0.0], [10.0, 0.0]), 0.0);
        assert_eq!(
            instance.gradient_between([1.0, 1.0], [1.0, 1.0]),
            0.0,
            "a zero-length segment has no defined slope"
        );
    }

    #[test]
    fn the_control_polygon_brackets_the_backbone_endpoints() {
        let instance = flat_instance();

        let polygon = instance.control_polygon(&[1.0, -1.0, 2.0]);

        assert_eq!(polygon.len(), 5);
        assert_eq!(polygon[0], instance.backbone_points[0]);
        assert_eq!(polygon[4], *instance.backbone_points.last().unwrap());
    }

    #[test]
    fn offsets_beyond_the_bounds_are_clamped() {
        let instance = flat_instance();

        let polygon = instance.control_polygon(&[1e6]);

        // The bound is +10 around a backbone that runs at column 20.
        assert_eq!(polygon[1][1], 30.0);
    }

    #[test]
    fn a_zero_offset_route_costs_the_backbone_length() {
        let instance = flat_instance();

        let value = instance.evaluate_solution(&[0.0; 5]);

        assert!(
            (value - instance.backbone_total_length).abs() < 1e-6,
            "a straight route on flat terrain costs its length, got {value}"
        );
    }

    #[test]
    fn the_objective_is_finite_for_every_allowed_dimension() {
        let instance = flat_instance();

        for &dimension in &TEST_DIMENSIONS {
            let value = instance.evaluate_solution(&vec![0.5; dimension as usize]);

            assert!(
                value.is_finite() && value >= 0.0,
                "D={dimension} gave {value}"
            );
        }
    }

    #[test]
    fn steep_terrain_is_more_expensive_than_flat_terrain() {
        let flat = flat_instance();
        let mut steep = flat_instance();
        for (i, mut row) in steep.heightmap.rows_mut().into_iter().enumerate() {
            row.fill(i as f64 * 10.0);
        }

        let route = [[0.0, 20.0], [10.0, 20.0], [20.0, 20.0]];

        assert!(
            steep.weighted_length(&route) > flat.weighted_length(&route),
            "the gradient penalty must apply"
        );
    }

    #[test]
    fn a_route_shorter_than_a_segment_has_no_length() {
        let instance = flat_instance();

        assert_eq!(instance.weighted_length(&[]), 0.0);
        assert_eq!(instance.weighted_length(&[[0.0, 0.0]]), 0.0);
    }
}
