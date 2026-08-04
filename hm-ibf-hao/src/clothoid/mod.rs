//! Asymmetric clothoid spline interpolation.
//!
//! Implements the controlled clothoid spline of D. J. Walton and D. S. Meek, "A controlled
//! clothoid spline", *Computers & Graphics* 29 (2005) 353-363. A route is described by a
//! polyline of control points; each consecutive triple is replaced by a pair of clothoid
//! arcs that meet tangentially, giving a curvature-continuous path.
//!
//! The `tau` parameter controls how much asymmetry between the two arcs is allowed
//! (`0.0` = symmetric, `1.0` = as asymmetric as the feasibility limit permits).
//!
//! The module is split by concern:
//!
//! * [`fresnel`] - the Fresnel integral approximation the clothoid is defined by.
//! * [`geometry`] - 2-D vector helpers and line reflection.
//! * [`solver`] - the joint solver that places the two arcs of a single triple.
//! * [`curve`] - construction of one asymmetric clothoid curve.
//! * [`spline`] - chaining curves into a full route.
//! * [`curvature`] - the numerical curvature constraint check.

pub mod curvature;
pub mod curve;
pub mod fresnel;
pub mod geometry;
pub mod solver;
pub mod spline;

pub use curvature::check_curvature_constraint_numerical;
pub use spline::interpolate_path_with_tau;

/// Step of the coarse scan over the joint angle, in radians.
///
/// The scan only has to bracket the minimum; [`solver`] refines it with a golden-section
/// search, so a coarse step costs accuracy nowhere and saves a large constant factor.
pub(crate) const JOINT_SCAN_STEP: f64 = 0.05;

/// Distance below which the joint is considered closed and refinement stops.
pub(crate) const JOINT_CLOSED_TOLERANCE: f64 = 1e-8;

/// Lengths below this are treated as zero when normalizing a direction.
pub(crate) const DEGENERATE_LENGTH: f64 = 1e-10;
