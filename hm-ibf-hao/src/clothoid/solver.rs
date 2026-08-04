//! Placement of the two clothoid arcs that join a triple of control points.
//!
//! Each triple `(P0, P1, P2)` is spanned by two clothoid arcs: one leaving `P0` towards
//! `P1` and one leaving `P2` towards `P1`. Their shapes are fixed by a single free
//! parameter `omega0`, the share of the total turn taken by the first arc. The solver
//! searches for the `omega0` that closes the gap between the two arc endpoints.

use std::f64::consts::PI;

use super::{
    fresnel,
    geometry::{self, Line, Point},
    JOINT_CLOSED_TOLERANCE,
};

/// Reciprocal of the golden ratio, the contraction factor of the refinement search.
const GOLDEN_INV: f64 = 0.618_033_988_749_894_9;

/// Width of the refinement bracket below which the search stops.
const BRACKET_TOLERANCE: f64 = 1e-10;

/// Minimum horizontal separation enforced between the two arc anchors.
///
/// The closed-form scale factor divides by the difference of the two anchor abscissae, so
/// a vanishing difference has to be nudged apart to keep the expression finite.
const MIN_ANCHOR_SEPARATION: f64 = 0.025;

/// One end of a joint: where an arc starts and how far it has turned.
#[derive(Clone, Copy, Debug)]
struct ArcEnd {
    /// The control point the arc emanates from.
    point: Point,
    /// Unit tangent pointing from the control point towards the corner.
    tangent: Point,
    /// Unit normal, the tangent rotated by a quarter turn.
    normal: Point,
    /// Fresnel cosine integral at this end's turn.
    cosine: f64,
    /// Fresnel sine integral at this end's turn.
    sine: f64,
    /// The turn taken by this arc, in radians.
    omega: f64,
    /// Normalized arc-length parameter at which this arc ends.
    parameter: f64,
}

impl ArcEnd {
    /// Builds an end from its geometry and turn.
    fn new(point: Point, tangent: Point, normal: Point, omega: f64) -> Self {
        let parameter = (2.0 * omega / PI).sqrt();
        let (cosine, sine) = fresnel::integrals(parameter);
        Self {
            point,
            tangent,
            normal,
            cosine,
            sine,
            omega,
            parameter,
        }
    }

    /// Returns the arc's endpoint for a given scale factor.
    fn endpoint(&self, scale: f64) -> Point {
        let along = scale * self.cosine;
        let across = scale * self.sine;
        [
            self.point[0] + along * self.tangent[0] + across * self.normal[0],
            self.point[1] + along * self.tangent[1] + across * self.normal[1],
        ]
    }
}

/// The fixed geometry of one joint, i.e. everything that does not depend on `omega0`.
#[derive(Clone, Copy, Debug)]
pub struct Joint {
    /// Angle enclosed by the two legs of the control polygon at the corner.
    pub omega: f64,
    /// The control point the first arc emanates from.
    pub point0: Point,
    /// Unit tangent of the first leg, pointing from `point0` towards the corner.
    pub tangent0: Point,
    /// Unit normal of the first leg.
    pub normal0: Point,
    /// The line carrying the first leg, used to mirror the arc when the corner turns left.
    pub line0: Line,
    /// The control point the second arc emanates from.
    pub point2: Point,
    /// Unit tangent of the second leg, pointing from `point2` towards the corner.
    pub tangent2: Point,
    /// Unit normal of the second leg.
    pub normal2: Point,
    /// The line carrying the second leg.
    pub line2: Line,
    /// Sign of the turn at the corner; positive means the polygon turns left.
    pub cross: f64,
}

/// A candidate arc pair and how far apart its two ends are.
#[derive(Clone, Copy, Debug)]
pub struct JointSolution {
    /// Distance between the endpoints of the two arcs.
    pub distance: f64,
    /// Normalized arc-length parameter of the first arc.
    pub parameter0: f64,
    /// Scale factor of the first arc.
    pub scale0: f64,
    /// Normalized arc-length parameter of the second arc.
    pub parameter2: f64,
    /// Scale factor of the second arc.
    pub scale2: f64,
}

impl JointSolution {
    /// Returns the solution reported when a candidate `omega0` admits no arc pair.
    fn unsolvable() -> Self {
        Self {
            distance: f64::INFINITY,
            parameter0: -1.0,
            scale0: -1.0,
            parameter2: -1.0,
            scale2: -1.0,
        }
    }

    /// Returns whether the two arcs meet closely enough to accept the solution.
    pub fn is_closed(&self) -> bool {
        self.distance < JOINT_CLOSED_TOLERANCE
    }

    /// Returns whether both arcs are non-degenerate and their ends nearly coincide.
    pub fn is_usable(&self) -> bool {
        self.distance <= 1.0 && self.parameter0 >= 1e-6 && self.parameter2 >= 1e-6
    }
}

impl Joint {
    /// Evaluates the arc pair produced by a given split of the total turn.
    ///
    /// # Arguments
    ///
    /// * `omega0` - Share of the turn assigned to the first arc, in radians.
    ///
    /// # Returns
    ///
    /// The resulting arc pair, or `None` if the split leaves the second arc a negative turn.
    fn evaluate(&self, omega0: f64) -> Option<JointSolution> {
        let omega2 = PI - self.omega - omega0;
        if omega0 < 0.0 || omega2 < 0.0 {
            return None;
        }

        let end0 = ArcEnd::new(self.point0, self.tangent0, self.normal0, omega0);
        let end2 = ArcEnd::new(self.point2, self.tangent2, self.normal2, omega2);

        // The closed-form scale factor is derived for the arc that gets mirrored; which of
        // the two that is depends on the turn direction, so the roles swap with `cross`.
        let (scale0, scale2) = if self.cross > 0.0 {
            let scale0 = anchor_scale(&end0, &end2, self.line0);
            let scale2 = if omega0 == 0.0 {
                f64::INFINITY
            } else {
                scale0 * (omega2 / omega0).sqrt()
            };
            (scale0, scale2)
        } else {
            let scale2 = anchor_scale(&end2, &end0, self.line2);
            (scale2 * (omega0 / omega2).sqrt(), scale2)
        };

        let mut p0 = end0.endpoint(scale0);
        if self.cross > 0.0 {
            p0 = self.line0.reflect(p0);
        }

        let distance = if scale2.is_infinite() {
            f64::INFINITY
        } else {
            let mut p2 = end2.endpoint(scale2);
            if self.cross < 0.0 {
                p2 = self.line2.reflect(p2);
            }
            geometry::length(geometry::from_to(p0, p2))
        };

        Some(JointSolution {
            distance,
            parameter0: end0.parameter,
            scale0,
            parameter2: end2.parameter,
            scale2,
        })
    }

    /// Evaluates a split, reporting an unsolvable solution instead of `None`.
    fn evaluate_or_unsolvable(&self, omega0: f64) -> JointSolution {
        self.evaluate(omega0)
            .unwrap_or_else(JointSolution::unsolvable)
    }

    /// Scans the admissible splits on a fixed grid and keeps the best.
    ///
    /// # Arguments
    ///
    /// * `range` - Inclusive-exclusive bounds of the scan.
    /// * `step` - Grid step; its sign selects the scan direction.
    ///
    /// # Returns
    ///
    /// The best solution found and the `omega0` that produced it.
    fn scan(&self, range: (f64, f64), step: f64) -> (JointSolution, f64) {
        let (start, end) = range;
        let mut best = JointSolution::unsolvable();
        let mut best_omega0 = start;

        let mut omega0 = start;
        while (step > 0.0 && omega0 < end) || (step < 0.0 && omega0 > end) {
            if let Some(candidate) = self.evaluate(omega0) {
                if candidate.distance < best.distance {
                    best = candidate;
                    best_omega0 = omega0;
                }
            }
            omega0 += step;
        }

        (best, best_omega0)
    }

    /// Finds the split of the turn that closes the joint.
    ///
    /// A coarse grid scan brackets the minimum and a golden-section search refines it,
    /// which reaches the same precision as a fine scan in a logarithmic number of steps.
    ///
    /// # Arguments
    ///
    /// * `range` - Inclusive-exclusive bounds of the scan.
    /// * `step` - Grid step of the coarse scan; its sign selects the scan direction.
    ///
    /// # Returns
    ///
    /// The best arc pair found. Callers must check [`JointSolution::is_usable`] before
    /// building a curve from it.
    pub fn solve(&self, range: (f64, f64), step: f64) -> JointSolution {
        let (mut best, best_omega0) = self.scan(range, step);
        if best.is_closed() {
            return best;
        }

        let (mut low, mut high) = (
            (best_omega0 - step.abs()).max(range.0.min(range.1)),
            (best_omega0 + step.abs()).min(range.0.max(range.1)),
        );
        let mut left = high - (high - low) * GOLDEN_INV;
        let mut right = low + (high - low) * GOLDEN_INV;
        let mut left_solution = self.evaluate_or_unsolvable(left);
        let mut right_solution = self.evaluate_or_unsolvable(right);

        for candidate in [left_solution, right_solution] {
            if candidate.distance < best.distance {
                best = candidate;
            }
        }

        while (high - low).abs() > BRACKET_TOLERANCE && !best.is_closed() {
            if left_solution.distance < right_solution.distance {
                high = right;
                right = left;
                right_solution = left_solution;
                left = high - (high - low) * GOLDEN_INV;
                left_solution = self.evaluate_or_unsolvable(left);
                if left_solution.distance < best.distance {
                    best = left_solution;
                }
            } else {
                low = left;
                left = right;
                left_solution = right_solution;
                right = low + (high - low) * GOLDEN_INV;
                right_solution = self.evaluate_or_unsolvable(right);
                if right_solution.distance < best.distance {
                    best = right_solution;
                }
            }
        }

        best
    }
}

/// Returns the scale factor placing the near arc's endpoint on the mirror line.
///
/// Solving "the reflected endpoint of the near arc lies on the far arc's ray" for the
/// scale factor gives a closed form whose branches follow the mirror line's orientation.
///
/// # Arguments
///
/// * `near` - The arc whose scale factor is solved for.
/// * `far` - The opposite arc, whose scale factor follows from the turn ratio.
/// * `line` - The leg the near arc is mirrored across.
///
/// # Returns
///
/// The scale factor, which may be infinite when the near arc takes no turn at all.
fn anchor_scale(near: &ArcEnd, far: &ArcEnd, line: Line) -> f64 {
    let near_x = near.point[0];
    let near_y = near.point[1];
    let near_dx = near.cosine * near.tangent[0] + near.sine * near.normal[0];
    let near_dy = near.cosine * near.tangent[1] + near.sine * near.normal[1];

    let far_dx = if near.omega == 0.0 {
        f64::INFINITY
    } else {
        (far.omega / near.omega).sqrt() * (far.cosine * far.tangent[0] + far.sine * far.normal[0])
    };

    let mut far_x = far.point[0];
    if (near_x - far_x).abs() < MIN_ANCHOR_SEPARATION {
        far_x = near_x + MIN_ANCHOR_SEPARATION;
    }

    if line.k.is_infinite() {
        (2.0 * line.n - near_x - far_x) / (far_dx + near_dx)
    } else if line.k == 0.0 {
        (near_x - far_x) / (far_dx - near_dx)
    } else {
        let k = line.k;
        (2.0 * k * near_y - 2.0 * k * line.n + near_x - near_x * k * k - far_x * k * k - far_x)
            / (-2.0 * k * near_dy - near_dx + near_dx * k * k + far_dx * k * k + far_dx)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use super::*;
    use crate::clothoid::JOINT_SCAN_STEP;

    /// Builds the joint of a symmetric right-angle corner at the origin.
    fn right_angle_joint() -> Joint {
        let point0 = [-10.0, 0.0];
        let corner = [0.0, 0.0];
        let point2 = [0.0, -10.0];
        // Tangents point from each control point towards the corner.
        let tangent0 = geometry::normalized(geometry::from_to(point0, corner)).unwrap();
        let tangent2 = geometry::normalized(geometry::from_to(point2, corner)).unwrap();

        Joint {
            omega: FRAC_PI_2,
            point0,
            tangent0,
            normal0: [-tangent0[1], tangent0[0]],
            line0: Line::through(point0, corner, 0.01),
            point2,
            tangent2,
            normal2: [-tangent2[1], tangent2[0]],
            line2: Line::through(point2, corner, 0.01),
            cross: geometry::cross(
                geometry::from_to(corner, point0),
                geometry::from_to(corner, point2),
            ),
        }
    }

    #[test]
    fn a_right_angle_corner_admits_a_closed_joint() {
        let solution = right_angle_joint().solve((0.0, PI - FRAC_PI_2), JOINT_SCAN_STEP);

        assert!(solution.is_usable(), "{solution:?}");
        assert!(solution.distance < 1e-6, "{solution:?}");
    }

    #[test]
    fn a_split_that_over_spends_the_turn_is_rejected() {
        let joint = right_angle_joint();

        assert!(joint.evaluate(-0.1).is_none());
        assert!(joint.evaluate(PI).is_none(), "omega2 would be negative");
    }

    #[test]
    fn refinement_never_worsens_the_coarse_scan() {
        let joint = right_angle_joint();
        let range = (0.0, PI - FRAC_PI_2);

        let (coarse, _) = joint.scan(range, JOINT_SCAN_STEP);
        let refined = joint.solve(range, JOINT_SCAN_STEP);

        assert!(
            refined.distance <= coarse.distance,
            "{refined:?} vs {coarse:?}"
        );
    }

    #[test]
    fn an_unsolvable_joint_is_reported_as_unusable() {
        let solution = JointSolution::unsolvable();

        assert!(!solution.is_usable());
        assert!(!solution.is_closed());
    }

    #[test]
    fn a_scan_over_an_empty_range_yields_no_solution() {
        let joint = right_angle_joint();

        let (solution, _) = joint.scan((0.5, 0.5), JOINT_SCAN_STEP);

        assert!(!solution.is_usable());
    }

    #[test]
    fn a_vanishing_first_turn_makes_the_second_scale_infinite() {
        let joint = right_angle_joint();

        let solution = joint.evaluate(0.0).unwrap();

        assert!(solution.scale2.is_infinite(), "{solution:?}");
        assert!(solution.distance.is_infinite(), "{solution:?}");
    }
}
