//! Construction of a single asymmetric clothoid curve through a triple of control points.

use std::f64::consts::PI;

use super::{
    fresnel,
    geometry::{self, Line, Point},
    solver::{Joint, JointSolution},
    DEGENERATE_LENGTH, JOINT_SCAN_STEP,
};

/// Horizontal separation below which a leg of the control polygon counts as vertical.
const VERTICAL_LEG_TOLERANCE: f64 = 0.01;

/// Relative slack around a straight or fully reversed corner that is treated as degenerate.
const STRAIGHT_CORNER_SLACK: f64 = 1e-6;

/// Builds the clothoid curve interpolating a triple of control points.
///
/// Degenerate triples - coincident points, a straight corner, a fully reversed corner, or a
/// joint the solver cannot close - fall back to the straight segment `[P0, P2]`, which keeps
/// the objective finite and monotone in the offsets rather than hiding the failure behind a
/// distorted curve.
///
/// # Arguments
///
/// * `control_points` - The triple `[P0, P1, P2]` to interpolate.
/// * `tau` - Asymmetry parameter in `[0, 1]`; `0` shortens an infeasible corner symmetrically.
/// * `point_step` - Step in the arc-length parameter between sampled output points.
///
/// # Returns
///
/// The sampled curve, always containing at least the two endpoints.
pub fn asymmetric_clothoid(control_points: &[Point; 3], tau: f64, point_step: f64) -> Vec<Point> {
    let [p0, p1, p2] = *control_points;
    let straight = vec![p0, p2];

    match corner_angle(&[p0, p1, p2]) {
        Some(omega) if !is_degenerate_corner(omega) => {}
        _ => return straight,
    }

    let (points, restored_start, restored_end) = make_feasible(control_points, tau);
    let Some(joint) = build_joint(&points) else {
        return straight;
    };

    let solution = joint.solve((0.0, PI - joint.omega), JOINT_SCAN_STEP);
    if !solution.is_usable() {
        return straight;
    }

    let mut curve = Vec::new();
    curve.extend(restored_start);

    let mirror_first = joint.cross > 0.0;
    let mut phi = 0.0;
    while phi < solution.parameter0 {
        curve.push(first_arc_point(&joint, phi, solution.scale0, mirror_first));
        phi += point_step;
    }

    // The two arcs only meet within the solver's tolerance, so the seam is bridged by the
    // midpoint of their two ends instead of emitting both.
    let seam = first_arc_point(&joint, solution.parameter0, solution.scale0, mirror_first);
    append_second_arc(&joint, &solution, point_step, seam, &mut curve);

    curve.extend(restored_end);
    curve
}

/// Returns the angle enclosed at the corner by the two legs of a control triple.
fn corner_angle(points: &[Point; 3]) -> Option<f64> {
    geometry::angle_between(
        geometry::from_to(points[1], points[0]),
        geometry::from_to(points[1], points[2]),
    )
}

/// Returns whether a corner is so straight or so reversed that no clothoid spans it.
fn is_degenerate_corner(omega: f64) -> bool {
    omega <= PI * STRAIGHT_CORNER_SLACK || omega >= PI * (1.0 - STRAIGHT_CORNER_SLACK)
}

/// Assembles the joint description of an already feasible control triple.
///
/// # Arguments
///
/// * `points` - The feasible triple `[P0, P1, P2]`.
///
/// # Returns
///
/// The joint, or `None` if a leg is degenerate.
fn build_joint(points: &[Point; 3]) -> Option<Joint> {
    let corner = points[1];
    let (tangent0, tangent2) = (
        geometry::normalized(geometry::from_to(points[0], corner))?,
        geometry::normalized(geometry::from_to(points[2], corner))?,
    );
    let omega = corner_angle(points)?;

    Some(Joint {
        omega,
        point0: points[0],
        tangent0,
        normal0: [-tangent0[1], tangent0[0]],
        line0: Line::through(points[0], corner, VERTICAL_LEG_TOLERANCE),
        point2: points[2],
        tangent2,
        normal2: [-tangent2[1], tangent2[0]],
        line2: Line::through(points[2], corner, VERTICAL_LEG_TOLERANCE),
        cross: geometry::cross(
            geometry::from_to(corner, points[0]),
            geometry::from_to(corner, points[2]),
        ),
    })
}

/// Returns a point of the first arc, mirrored across its leg when the corner turns left.
fn first_arc_point(joint: &Joint, phi: f64, scale: f64, mirror: bool) -> Point {
    let point = clothoid_point(phi, joint.point0, joint.tangent0, joint.normal0, scale);
    if mirror {
        joint.line0.reflect(point)
    } else {
        point
    }
}

/// Samples the second arc backwards, from the seam down to its control point.
///
/// # Arguments
///
/// * `joint` - The joint the arc belongs to.
/// * `solution` - The solved joint, supplying the second arc's parameter and scale.
/// * `point_step` - Step in the arc-length parameter between sampled points.
/// * `seam` - Endpoint of the first arc, blended into the second arc's first sample.
/// * `out` - Destination the samples are appended to.
fn append_second_arc(
    joint: &Joint,
    solution: &JointSolution,
    point_step: f64,
    seam: Point,
    out: &mut Vec<Point>,
) {
    let mirror = joint.cross < 0.0;
    let sample = |phi: f64| {
        let point = clothoid_point(
            phi,
            joint.point2,
            joint.tangent2,
            joint.normal2,
            solution.scale2,
        );
        if mirror {
            joint.line2.reflect(point)
        } else {
            point
        }
    };

    let mut first = true;
    let mut phi = solution.parameter2;
    while phi > 0.0 {
        let point = sample(phi);
        out.push(if first {
            first = false;
            geometry::midpoint(point, seam)
        } else {
            point
        });
        phi -= point_step;
    }

    out.push(sample(0.0));
}

/// Returns the point at arc-length parameter `phi` on a clothoid.
///
/// # Arguments
///
/// * `phi` - Normalized arc-length parameter, `0` at the origin.
/// * `origin` - The point the clothoid starts from.
/// * `tangent` - Unit tangent at the origin.
/// * `normal` - Unit normal at the origin.
/// * `scale` - Scale factor of the clothoid.
///
/// # Returns
///
/// The curve point.
fn clothoid_point(phi: f64, origin: Point, tangent: Point, normal: Point, scale: f64) -> Point {
    let (cosine, sine) = fresnel::integrals(phi);
    let (along, across) = (scale * cosine, scale * sine);
    [
        origin[0] + along * tangent[0] + across * normal[0],
        origin[1] + along * tangent[1] + across * normal[1],
    ]
}

/// Pulls the longer leg of the control polygon in until the corner admits a clothoid pair.
///
/// A corner is only spannable when the two legs are not too unequal. When they are, the
/// longer leg's control point is moved towards the corner; `tau` interpolates between
/// matching the shorter leg (`0`) and the longest still-feasible leg (`1`). The point that
/// was moved is returned so the caller can re-attach the original vertex to the curve.
///
/// # Arguments
///
/// * `control_points` - The original triple.
/// * `tau` - Asymmetry parameter in `[0, 1]`.
///
/// # Returns
///
/// The adjusted triple, plus the original first and last points when they were moved.
fn make_feasible(
    control_points: &[Point; 3],
    tau: f64,
) -> ([Point; 3], Option<Point>, Option<Point>) {
    let mut points = *control_points;
    let corner = points[1];

    let (Some(tangent0), Some(tangent2)) = (
        geometry::normalized(geometry::from_to(points[0], corner)),
        geometry::normalized(geometry::from_to(points[2], corner)),
    ) else {
        return (points, None, None);
    };
    let length0 = geometry::length(geometry::from_to(points[0], corner));
    let length2 = geometry::length(geometry::from_to(points[2], corner));
    if length0 < DEGENERATE_LENGTH || length2 < DEGENERATE_LENGTH {
        return (points, None, None);
    }

    let Some(omega) = corner_angle(&points) else {
        return (points, None, None);
    };
    let alpha = PI - omega;
    let (cosine, sine) = fresnel::integrals((2.0 * alpha / PI).sqrt());
    let limit = cosine / sine;

    // `ratio >= limit` is the infeasibility test of Walton & Meek: beyond it the two arcs
    // can no longer be scaled to meet, so the longer leg has to give.
    if length0 > length2 {
        let ratio = (length0 / length2 + alpha.cos()) / alpha.sin();
        if ratio >= limit {
            let feasible = length2 * (limit * alpha.sin() - alpha.cos());
            let distance = (1.0 - tau) * length2 + tau * feasible;
            points[0] = step_from(corner, tangent0, distance);
            return (points, Some(control_points[0]), None);
        }
    } else if length2 > length0 {
        let ratio = (length2 / length0 + alpha.cos()) / alpha.sin();
        if ratio >= limit {
            let feasible = length0 * (limit * alpha.sin() - alpha.cos());
            let distance = (1.0 - tau) * length0 + tau * feasible;
            points[2] = step_from(corner, tangent2, distance);
            return (points, None, Some(control_points[2]));
        }
    }

    (points, None, None)
}

/// Returns the point `distance` back from `corner` along `toward_corner`.
#[inline]
fn step_from(corner: Point, toward_corner: Point, distance: f64) -> Point {
    [
        corner[0] - distance * toward_corner[0],
        corner[1] - distance * toward_corner[1],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: f64 = 0.05;

    /// The leg length from a control point to the corner.
    fn leg(points: &[Point; 3], index: usize) -> f64 {
        geometry::length(geometry::from_to(points[index], points[1]))
    }

    #[test]
    fn a_right_angle_corner_is_rounded() {
        let curve = asymmetric_clothoid(&[[-10.0, 0.0], [0.0, 0.0], [0.0, -10.0]], 0.4, STEP);

        assert!(curve.len() > 2, "expected a sampled curve, got {curve:?}");
        assert!(curve.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
    }

    #[test]
    fn a_straight_corner_falls_back_to_the_chord() {
        let curve = asymmetric_clothoid(&[[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], 0.4, STEP);

        assert_eq!(curve, vec![[0.0, 0.0], [10.0, 0.0]]);
    }

    #[test]
    fn a_fully_reversed_corner_falls_back_to_the_chord() {
        let curve = asymmetric_clothoid(&[[0.0, 0.0], [5.0, 0.0], [0.0, 0.0]], 0.4, STEP);

        assert_eq!(curve, vec![[0.0, 0.0], [0.0, 0.0]]);
    }

    #[test]
    fn a_coincident_triple_falls_back_to_the_chord() {
        let curve = asymmetric_clothoid(&[[1.0, 1.0], [1.0, 1.0], [4.0, 4.0]], 0.4, STEP);

        assert_eq!(curve, vec![[1.0, 1.0], [4.0, 4.0]]);
    }

    #[test]
    fn both_turn_directions_produce_a_curve() {
        let left = asymmetric_clothoid(&[[-10.0, 0.0], [0.0, 0.0], [0.0, -10.0]], 0.4, STEP);
        let right = asymmetric_clothoid(&[[-10.0, 0.0], [0.0, 0.0], [0.0, 10.0]], 0.4, STEP);

        assert!(left.len() > 2);
        assert!(right.len() > 2);
    }

    #[test]
    fn the_curve_starts_and_ends_near_the_outer_control_points() {
        let points = [[-10.0, 0.0], [0.0, 0.0], [0.0, -10.0]];

        let curve = asymmetric_clothoid(&points, 0.4, STEP);

        // The endpoints are evaluated at arc-length parameter zero, where the Fresnel
        // approximation returns zero only up to rounding, so a tolerance is required.
        assert!(
            geometry::length(geometry::from_to(curve[0], points[0])) < 1e-9,
            "the first arc starts at P0, got {:?}",
            curve[0]
        );
        let last = *curve.last().unwrap();
        assert!(
            geometry::length(geometry::from_to(last, points[2])) < 1e-9,
            "the second arc ends at P2, got {last:?}"
        );
    }

    #[test]
    fn an_infeasible_corner_keeps_its_original_far_vertex() {
        let far = [0.0, -400.0];

        let curve = asymmetric_clothoid(&[[-10.0, 0.0], [0.0, 0.0], far], 1.0, STEP);

        assert_eq!(
            *curve.last().unwrap(),
            far,
            "the moved vertex is re-attached"
        );
    }

    #[test]
    fn making_a_balanced_corner_feasible_leaves_it_untouched() {
        let original = [[-10.0, 0.0], [0.0, 0.0], [0.0, -10.0]];

        let (points, start, end) = make_feasible(&original, 0.4);

        assert_eq!(points, original);
        assert!(start.is_none() && end.is_none());
    }

    #[test]
    fn making_an_infeasible_corner_feasible_shortens_the_longer_leg() {
        let original = [[-400.0, 0.0], [0.0, 0.0], [0.0, -10.0]];

        let (points, start, end) = make_feasible(&original, 1.0);

        assert_eq!(start, Some(original[0]), "the first leg is the longer one");
        assert!(end.is_none());
        assert!(leg(&points, 0) < leg(&original, 0), "{points:?}");
        assert_eq!(points[2], original[2], "the short leg is untouched");
    }

    #[test]
    fn the_shortened_leg_stays_on_the_original_ray() {
        let original = [[-400.0, 0.0], [0.0, 0.0], [0.0, -10.0]];

        let (points, _, _) = make_feasible(&original, 1.0);

        assert!(
            points[0][0] < 0.0,
            "P0 must stay on its own side: {points:?}"
        );
        assert!(
            points[0][1].abs() < 1e-9,
            "P0 must stay on the ray: {points:?}"
        );
    }

    #[test]
    fn a_smaller_tau_shortens_the_long_leg_further() {
        let original = [[-400.0, 0.0], [0.0, 0.0], [0.0, -10.0]];

        let (symmetric, _, _) = make_feasible(&original, 0.0);
        let (asymmetric, _, _) = make_feasible(&original, 1.0);

        assert!(
            leg(&symmetric, 0) < leg(&asymmetric, 0),
            "{} vs {}",
            leg(&symmetric, 0),
            leg(&asymmetric, 0)
        );
    }
}
