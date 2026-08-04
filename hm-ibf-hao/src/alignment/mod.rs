//! The horizontal alignment optimization problem.
//!
//! An instance fixes a *backbone*: an equidistant resampling of a least-cost route across a
//! terrain heightmap. A solution is a vector of `D` perpendicular offsets applied to the
//! backbone, one per control point. The offset points are interpolated with an asymmetric
//! clothoid spline and the resulting route is scored as terrain-weighted length plus a
//! penalty for every sample that turns tighter than the minimum radius of curvature.
//!
//! `D` is not fixed by the instance: island metaheuristics run at different dimensions and
//! control points are placed at `s_i = i·L/(D+1)` along the backbone, so every dimension
//! describes the same route at a different resolution and their objective values are
//! directly comparable.

pub mod config;
pub mod evaluator;
pub mod output;
pub mod problem;
pub mod samples;
#[cfg(test)]
pub(crate) mod test_support;

pub use config::{AlignmentConfig, BackboneConfig};
pub use evaluator::AlignmentEvaluator;
pub use output::{write_run_results, RunMetadata, OUTPUT_EXPORT_SCHEMA_VERSION};
pub use problem::HorizontalAlignment;
pub use samples::BackboneSamples;

/// Weight applied to each sample that violates the minimum radius of curvature.
///
/// Curvature violations make a route unbuildable rather than merely expensive, so the weight
/// is far above any plausible length difference and the search treats them as hard.
pub const CURVATURE_PENALTY: f64 = 1000.0;

/// Step in the arc-length parameter at which the clothoid route is sampled.
///
/// Fixed across dimensions so that the sampled length - and therefore the objective value -
/// is comparable between islands that run at different `D`.
pub const CLOTHOID_POINT_STEP: f64 = 0.01;
