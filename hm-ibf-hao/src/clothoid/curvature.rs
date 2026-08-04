//! Numerical curvature constraint check for a sampled route.

use std::cmp::Ordering;

use super::geometry::Point;

/// Counts the sampled points whose radius of curvature falls below the allowed minimum.
///
/// The curvature is evaluated from finite differences of the sampled route,
/// `k = |x' y'' - y' x''| / (x'² + y'²)^{3/2}`, and the radius is its reciprocal. Points on
/// a straight stretch have zero curvature and an infinite radius, so they never violate. A
/// point whose curvature is indeterminate - a stationary sample, where every difference
/// vanishes - counts as a violation rather than being silently accepted.
///
/// # Arguments
///
/// * `points` - The sampled route.
/// * `min_radius` - Smallest radius of curvature the route may have.
///
/// # Returns
///
/// The number of violating points, as a float so callers can weight it directly into an
/// objective value.
pub fn check_curvature_constraint_numerical(points: &[Point], min_radius: f64) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    let xs: Vec<f64> = points.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = points.iter().map(|p| p[1]).collect();
    let dx = gradient(&xs);
    let dy = gradient(&ys);
    let ddx = gradient(&dx);
    let ddy = gradient(&dy);

    (0..points.len())
        .filter(|&i| {
            let numerator = (dx[i] * ddy[i] - dy[i] * ddx[i]).abs();
            let denominator = (dx[i] * dx[i] + dy[i] * dy[i]).powf(1.5);
            let curvature = numerator / denominator;
            let radius = if curvature == 0.0 {
                f64::INFINITY
            } else {
                1.0 / curvature
            };
            // Anything that is not strictly wider than the limit violates it, including an
            // indeterminate radius, which compares equal to nothing at all.
            !matches!(radius.partial_cmp(&min_radius), Some(Ordering::Greater))
        })
        .count() as f64
}

/// Returns the central-difference derivative of a series sampled at unit spacing.
///
/// Interior samples use the symmetric difference; the two ends fall back to the one-sided
/// difference, matching the behaviour of `numpy.gradient`.
///
/// # Arguments
///
/// * `values` - The sampled series.
///
/// # Returns
///
/// The derivative, of the same length as the input.
fn gradient(values: &[f64]) -> Vec<f64> {
    match values.len() {
        0 => return Vec::new(),
        1 => return vec![0.0],
        _ => {}
    }

    let n = values.len();
    let mut derivative = vec![0.0; n];
    derivative[0] = values[1] - values[0];
    derivative[n - 1] = values[n - 1] - values[n - 2];
    for i in 1..n - 1 {
        derivative[i] = (values[i + 1] - values[i - 1]) * 0.5;
    }
    derivative
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;

    /// Samples a circle of the given radius at unit arc-length spacing.
    fn circle(radius: f64, samples: usize) -> Vec<Point> {
        (0..samples)
            .map(|i| {
                let angle = TAU * i as f64 / samples as f64;
                [radius * angle.cos(), radius * angle.sin()]
            })
            .collect()
    }

    #[test]
    fn an_empty_route_has_no_violations() {
        assert_eq!(check_curvature_constraint_numerical(&[], 10.0), 0.0);
    }

    #[test]
    fn a_straight_route_never_violates() {
        let line: Vec<Point> = (0..50).map(|i| [i as f64, 0.0]).collect();

        assert_eq!(check_curvature_constraint_numerical(&line, 1e9), 0.0);
    }

    #[test]
    fn a_circle_tighter_than_the_limit_violates_everywhere() {
        let tight = circle(5.0, 200);

        let violations = check_curvature_constraint_numerical(&tight, 1000.0);

        assert_eq!(violations, tight.len() as f64);
    }

    #[test]
    fn a_circle_wider_than_the_limit_never_violates() {
        let wide = circle(100.0, 400);

        let violations = check_curvature_constraint_numerical(&wide, 1.0);

        assert_eq!(violations, 0.0);
    }

    #[test]
    fn a_stationary_sample_counts_as_a_violation() {
        let stalled = vec![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];

        let violations = check_curvature_constraint_numerical(&stalled, 10.0);

        assert_eq!(
            violations,
            stalled.len() as f64,
            "0/0 must not pass silently"
        );
    }

    #[test]
    fn the_gradient_of_a_ramp_is_its_slope() {
        let ramp: Vec<f64> = (0..10).map(|i| 3.0 * i as f64).collect();

        let derivative = gradient(&ramp);

        assert!(
            derivative.iter().all(|d| (d - 3.0).abs() < 1e-12),
            "{derivative:?}"
        );
    }

    #[test]
    fn degenerate_series_have_a_defined_gradient() {
        assert!(gradient(&[]).is_empty());
        assert_eq!(gradient(&[7.0]), vec![0.0]);
        assert_eq!(gradient(&[1.0, 4.0]), vec![3.0, 3.0]);
    }
}
