//! Chaining of single clothoid curves into a route through arbitrarily many control points.

use super::{curve::asymmetric_clothoid, geometry::Point};

/// Interpolates a route through control points with asymmetric clothoid splines.
///
/// Consecutive duplicates are dropped first; routes of fewer than three distinct points have
/// no corner to round and are returned unchanged. Longer routes are split into overlapping
/// triples joined at the midpoints of the intermediate legs, so each corner is rounded
/// independently while the arcs still meet.
///
/// # Arguments
///
/// * `control_points` - The route's control polygon.
/// * `tau` - Asymmetry parameter in `[0, 1]`.
/// * `point_step` - Step in the arc-length parameter between sampled output points.
///
/// # Returns
///
/// The sampled route.
pub fn interpolate_path_with_tau(
    control_points: &[Point],
    tau: f64,
    point_step: f64,
) -> Vec<Point> {
    let points = deduplicate(control_points);

    match points.len() {
        0..=2 => points,
        3 => asymmetric_clothoid(&[points[0], points[1], points[2]], tau, point_step),
        _ => chain_triples(&insert_midpoints(&points), tau, point_step),
    }
}

/// Removes consecutive duplicate points.
///
/// # Arguments
///
/// * `points` - The polyline to clean.
///
/// # Returns
///
/// The polyline without consecutive repetitions.
fn deduplicate(points: &[Point]) -> Vec<Point> {
    let mut result: Vec<Point> = Vec::with_capacity(points.len());
    for &point in points {
        if result.last() != Some(&point) {
            result.push(point);
        }
    }
    result
}

/// Splits every intermediate leg at its midpoint.
///
/// The midpoints become the shared endpoints of neighbouring triples, which is what lets
/// each corner be rounded on its own without the arcs of two corners overlapping.
///
/// # Arguments
///
/// * `points` - A polyline of at least four distinct points.
///
/// # Returns
///
/// The polyline with a midpoint inserted after every interior vertex but the last.
fn insert_midpoints(points: &[Point]) -> Vec<Point> {
    let mut result = vec![points[0]];

    for i in 1..points.len() - 2 {
        let current = points[i];
        let next = points[i + 1];
        result.push(current);
        result.push([
            current[0] + 0.5 * (next[0] - current[0]),
            current[1] + 0.5 * (next[1] - current[1]),
        ]);
    }

    result.extend_from_slice(&points[points.len() - 2..]);
    result
}

/// Rounds every corner of a midpoint-split polyline and concatenates the results.
///
/// # Arguments
///
/// * `points` - The midpoint-split polyline.
/// * `tau` - Asymmetry parameter in `[0, 1]`.
/// * `point_step` - Step in the arc-length parameter between sampled output points.
///
/// # Returns
///
/// The concatenated route.
fn chain_triples(points: &[Point], tau: f64, point_step: f64) -> Vec<Point> {
    let mut route = Vec::new();

    // Triples overlap in their shared endpoint, so the window advances by two.
    let mut start = 0;
    while start + 2 < points.len() {
        let triple = [points[start], points[start + 1], points[start + 2]];
        route.extend(asymmetric_clothoid(&triple, tau, point_step));
        start += 2;
    }

    route
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: f64 = 0.05;

    #[test]
    fn consecutive_duplicates_are_dropped() {
        let points = [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]];

        let cleaned = deduplicate(&points);

        assert_eq!(cleaned, vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]);
    }

    #[test]
    fn non_consecutive_repetitions_are_kept() {
        let points = [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]];

        assert_eq!(deduplicate(&points).len(), 3);
    }

    #[test]
    fn an_empty_route_stays_empty() {
        assert!(interpolate_path_with_tau(&[], 0.4, STEP).is_empty());
    }

    #[test]
    fn a_two_point_route_is_returned_unchanged() {
        let points = [[0.0, 0.0], [10.0, 5.0]];

        assert_eq!(
            interpolate_path_with_tau(&points, 0.4, STEP),
            points.to_vec()
        );
    }

    #[test]
    fn a_route_that_collapses_to_one_point_is_returned_unchanged() {
        let points = [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]];

        assert_eq!(
            interpolate_path_with_tau(&points, 0.4, STEP),
            vec![[3.0, 3.0]]
        );
    }

    #[test]
    fn a_three_point_route_is_a_single_clothoid() {
        let points = [[-10.0, 0.0], [0.0, 0.0], [0.0, -10.0]];

        let route = interpolate_path_with_tau(&points, 0.4, STEP);

        assert_eq!(
            route,
            asymmetric_clothoid(&[points[0], points[1], points[2]], 0.4, STEP)
        );
    }

    #[test]
    fn midpoints_are_inserted_between_interior_vertices() {
        let points = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [20.0, 10.0]];

        let split = insert_midpoints(&points);

        assert_eq!(
            split,
            vec![
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 5.0],
                [10.0, 10.0],
                [20.0, 10.0]
            ]
        );
    }

    #[test]
    fn a_longer_route_is_sampled_and_finite() {
        let points = [
            [0.0, 0.0],
            [10.0, 2.0],
            [20.0, -3.0],
            [30.0, 4.0],
            [40.0, 0.0],
        ];

        let route = interpolate_path_with_tau(&points, 0.4, STEP);

        assert!(
            route.len() > points.len(),
            "route has {} points",
            route.len()
        );
        assert!(route.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
    }

    #[test]
    fn every_corner_of_a_long_route_contributes_a_curve() {
        let points = [
            [0.0, 0.0],
            [10.0, 2.0],
            [20.0, -3.0],
            [30.0, 4.0],
            [40.0, 0.0],
            [50.0, 6.0],
        ];

        let split = insert_midpoints(&points);
        let route = chain_triples(&split, 0.4, STEP);

        // Triples advance two points at a time over the split polyline.
        assert_eq!(split.len(), 2 * (points.len() - 2) + 1);
        assert!(!route.is_empty());
    }

    #[test]
    fn a_straight_route_stays_on_its_line() {
        let points = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]];

        let route = interpolate_path_with_tau(&points, 0.4, STEP);

        assert!(
            route.iter().all(|p| p[1].abs() < 1e-9),
            "a straight route must not bend: {route:?}"
        );
    }
}
