//! Planar vector and line helpers shared by the clothoid construction.

use super::DEGENERATE_LENGTH;

/// A 2-D point or vector in heightmap pixel coordinates.
pub type Point = [f64; 2];

/// A line in the slope-intercept form `y = k·x + n`.
///
/// A vertical line is represented by an infinite `k`, in which case `n` holds its `x`
/// coordinate instead of the intercept.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    /// The slope, or infinity for a vertical line.
    pub k: f64,
    /// The `y` intercept, or the `x` coordinate of a vertical line.
    pub n: f64,
}

impl Line {
    /// Builds the line through two points.
    ///
    /// Segments that are vertical within `tolerance` are reported as vertical, which keeps
    /// the slope from exploding on nearly vertical control polygons.
    ///
    /// # Arguments
    ///
    /// * `from` - The first point on the line.
    /// * `to` - The second point on the line.
    /// * `tolerance` - Horizontal distance below which the line counts as vertical.
    ///
    /// # Returns
    ///
    /// The line through both points.
    pub fn through(from: Point, to: Point, tolerance: f64) -> Self {
        if (to[0] - from[0]).abs() < tolerance {
            return Self {
                k: f64::INFINITY,
                n: from[0],
            };
        }
        let k = (to[1] - from[1]) / (to[0] - from[0]);
        Self {
            k,
            n: from[1] - k * from[0],
        }
    }

    /// Reflects a point across this line.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to mirror.
    ///
    /// # Returns
    ///
    /// The mirrored point.
    pub fn reflect(&self, point: Point) -> Point {
        let (x_cross, y_cross) = if self.k.is_infinite() {
            (self.n, point[1])
        } else if self.k.abs() < DEGENERATE_LENGTH {
            (point[0], self.n)
        } else {
            let k_orthogonal = -1.0 / self.k;
            let n_orthogonal = point[1] - k_orthogonal * point[0];
            let x = (n_orthogonal - self.n) / (self.k - k_orthogonal);
            (x, self.k * x + self.n)
        };

        [
            x_cross + (x_cross - point[0]),
            y_cross + (y_cross - point[1]),
        ]
    }
}

/// Returns the euclidean length of a vector.
#[inline]
pub fn length(v: Point) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Returns the dot product of two vectors.
#[inline]
pub fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// Returns the scalar cross product of two vectors, i.e. the signed area of their span.
#[inline]
pub fn cross(a: Point, b: Point) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// Returns the vector pointing from `a` to `b`.
#[inline]
pub fn from_to(a: Point, b: Point) -> Point {
    [b[0] - a[0], b[1] - a[1]]
}

/// Returns the midpoint of two points.
#[inline]
pub fn midpoint(a: Point, b: Point) -> Point {
    [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
}

/// Returns `v` scaled to unit length, or `None` if it is shorter than [`DEGENERATE_LENGTH`].
#[inline]
pub fn normalized(v: Point) -> Option<Point> {
    let len = length(v);
    (len >= DEGENERATE_LENGTH).then(|| [v[0] / len, v[1] / len])
}

/// Returns the angle in `[0, π]` enclosed by two vectors.
///
/// # Arguments
///
/// * `a` - The first vector.
/// * `b` - The second vector.
///
/// # Returns
///
/// The enclosed angle, or `None` if either vector is degenerate.
pub fn angle_between(a: Point, b: Point) -> Option<f64> {
    let denominator = length(a) * length(b);
    (denominator >= DEGENERATE_LENGTH).then(|| (dot(a, b) / denominator).clamp(-1.0, 1.0).acos())
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use super::*;

    #[test]
    fn the_length_of_a_three_four_five_triangle_is_five() {
        assert_eq!(length([3.0, 4.0]), 5.0);
    }

    #[test]
    fn orthogonal_vectors_have_a_zero_dot_product() {
        assert_eq!(dot([1.0, 0.0], [0.0, 1.0]), 0.0);
        assert_eq!(cross([1.0, 0.0], [0.0, 1.0]), 1.0);
    }

    #[test]
    fn from_to_and_midpoint_are_consistent() {
        assert_eq!(from_to([1.0, 1.0], [4.0, 5.0]), [3.0, 4.0]);
        assert_eq!(midpoint([0.0, 0.0], [4.0, 2.0]), [2.0, 1.0]);
    }

    #[test]
    fn a_degenerate_vector_has_no_direction() {
        assert!(normalized([0.0, 0.0]).is_none());
        assert_eq!(normalized([0.0, 2.0]), Some([0.0, 1.0]));
    }

    #[test]
    fn angles_are_measured_in_the_unsigned_range() {
        let right = angle_between([1.0, 0.0], [0.0, 1.0]).unwrap();
        let straight = angle_between([1.0, 0.0], [-1.0, 0.0]).unwrap();

        assert!((right - FRAC_PI_2).abs() < 1e-12);
        assert!((straight - PI).abs() < 1e-12);
        assert!(angle_between([0.0, 0.0], [1.0, 1.0]).is_none());
    }

    #[test]
    fn reflecting_twice_returns_the_original_point() {
        for line in [
            Line::through([0.0, 0.0], [1.0, 1.0], 0.01),
            Line::through([0.0, 3.0], [5.0, 3.0], 0.01),
            Line::through([2.0, 0.0], [2.0, 7.0], 0.01),
        ] {
            let point = [4.0, -1.5];

            let round_trip = line.reflect(line.reflect(point));

            assert!((round_trip[0] - point[0]).abs() < 1e-9, "{round_trip:?}");
            assert!((round_trip[1] - point[1]).abs() < 1e-9, "{round_trip:?}");
        }
    }

    #[test]
    fn a_vertical_segment_yields_a_vertical_line() {
        let line = Line::through([2.0, 0.0], [2.0, 7.0], 0.01);

        assert!(line.k.is_infinite());
        assert_eq!(line.n, 2.0);
        assert_eq!(line.reflect([5.0, 1.0]), [-1.0, 1.0]);
    }

    #[test]
    fn a_point_on_the_line_is_its_own_reflection() {
        let line = Line::through([0.0, 1.0], [2.0, 3.0], 0.01);

        let reflected = line.reflect([1.0, 2.0]);

        assert!((reflected[0] - 1.0).abs() < 1e-9, "{reflected:?}");
        assert!((reflected[1] - 2.0).abs() < 1e-9, "{reflected:?}");
    }
}
