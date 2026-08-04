//! One-dimensional interpolation kernels used to resample an offset vector.

/// Values closer together than this are treated as coincident.
const EPSILON: f64 = 1e-10;

/// Pivots smaller than this are treated as zero when solving a tridiagonal system.
const PIVOT_EPSILON: f64 = 1e-14;

/// Computes the shape-preserving PCHIP slopes of a sampled series.
///
/// The slopes are chosen so the interpolant never overshoots between samples, which keeps a
/// resampled offset vector inside the range its source spanned.
///
/// # Arguments
///
/// * `x` - The sample abscissae, strictly ascending.
/// * `y` - The sample values, of the same length as `x`.
///
/// # Returns
///
/// One slope per sample.
pub fn pchip_derivatives(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let widths: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
    let slopes: Vec<f64> = (0..n - 1)
        .map(|i| {
            if widths[i].abs() > EPSILON {
                (y[i + 1] - y[i]) / widths[i]
            } else {
                0.0
            }
        })
        .collect();

    let mut derivatives = vec![0.0; n];
    if n == 2 {
        derivatives[0] = slopes[0];
        derivatives[1] = slopes[0];
        return derivatives;
    }

    for i in 1..n - 1 {
        derivatives[i] =
            weighted_harmonic_slope(widths[i - 1], widths[i], slopes[i - 1], slopes[i]);
    }
    derivatives[0] = end_derivative(widths[0], widths[1], slopes[0], slopes[1]);
    derivatives[n - 1] = end_derivative(widths[n - 2], widths[n - 3], slopes[n - 2], slopes[n - 3]);

    derivatives
}

/// Evaluates a PCHIP interpolant.
///
/// # Arguments
///
/// * `x` - The sample abscissae, strictly ascending.
/// * `y` - The sample values.
/// * `derivatives` - The slopes returned by [`pchip_derivatives`].
/// * `t` - The abscissa to evaluate at; clamped into the sampled range.
///
/// # Returns
///
/// The interpolated value.
pub fn pchip_evaluate(x: &[f64], y: &[f64], derivatives: &[f64], t: f64) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return y[0];
    }

    let t = t.clamp(x[0], x[n - 1]);
    let i = segment_index(x, t);
    let width = x[i + 1] - x[i];
    if width.abs() < EPSILON {
        return y[i];
    }

    let s = (t - x[i]) / width;
    let (s2, s3) = (s * s, s * s * s);
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;

    h00 * y[i] + h10 * width * derivatives[i] + h01 * y[i + 1] + h11 * width * derivatives[i + 1]
}

/// Computes the piecewise-cubic coefficients of an Akima spline.
///
/// # Arguments
///
/// * `x` - The sample abscissae, strictly ascending.
/// * `y` - The sample values.
///
/// # Returns
///
/// The `[a, b, c, d]` coefficients of each segment, empty for fewer than two samples.
pub fn akima_coefficients(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    let n = x.len();
    if n < 2 {
        return Vec::new();
    }

    let slopes = akima_slopes(x, y);

    (0..n - 1)
        .map(|i| {
            let width = x[i + 1] - x[i];
            if width.abs() < EPSILON {
                return [y[i], 0.0, 0.0, 0.0];
            }
            let delta = (y[i + 1] - y[i]) / width;
            [
                y[i],
                slopes[i],
                (3.0 * delta - 2.0 * slopes[i] - slopes[i + 1]) / width,
                (slopes[i] + slopes[i + 1] - 2.0 * delta) / (width * width),
            ]
        })
        .collect()
}

/// Evaluates an Akima spline.
///
/// # Arguments
///
/// * `x` - The sample abscissae, strictly ascending.
/// * `y` - The sample values.
/// * `coefficients` - The coefficients returned by [`akima_coefficients`].
/// * `t` - The abscissa to evaluate at; clamped into the sampled range.
///
/// # Returns
///
/// The interpolated value.
pub fn akima_evaluate(x: &[f64], y: &[f64], coefficients: &[[f64; 4]], t: f64) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 || coefficients.is_empty() {
        return y[0];
    }

    let t = t.clamp(x[0], x[n - 1]);
    let i = segment_index(x, t);
    let d = t - x[i];
    let c = &coefficients[i];

    c[0] + c[1] * d + c[2] * d * d + c[3] * d * d * d
}

/// Solves a tridiagonal system with the Thomas algorithm.
///
/// # Arguments
///
/// * `diagonal` - The main diagonal, of length `n`.
/// * `upper` - The superdiagonal, of length `n - 1`.
/// * `lower` - The subdiagonal, of length `n - 1`.
/// * `rhs` - The right-hand side, of length `n`.
///
/// # Returns
///
/// The solution vector; entries whose pivot vanished are left at zero.
pub fn tridiagonal_thomas(diagonal: &[f64], upper: &[f64], lower: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diagonal.len();
    let mut right = rhs.to_vec();
    let mut pivots = diagonal.to_vec();

    for i in 1..n {
        if pivots[i - 1].abs() < PIVOT_EPSILON {
            continue;
        }
        let factor = lower[i - 1] / pivots[i - 1];
        pivots[i] -= factor * upper[i - 1];
        right[i] -= factor * right[i - 1];
    }

    let mut solution = vec![0.0; n];
    if pivots[n - 1].abs() > PIVOT_EPSILON {
        solution[n - 1] = right[n - 1] / pivots[n - 1];
    }
    for i in (0..n - 1).rev() {
        if pivots[i].abs() > PIVOT_EPSILON {
            solution[i] = (right[i] - upper[i] * solution[i + 1]) / pivots[i];
        }
    }

    solution
}

/// Evaluates a natural-form cubic spline from its second derivatives.
///
/// # Arguments
///
/// * `x` - The sample abscissae, strictly ascending.
/// * `y` - The sample values.
/// * `moments` - The second derivatives at the samples.
/// * `widths` - The segment widths, of length `x.len() - 1`.
/// * `t` - The abscissa to evaluate at; clamped into the sampled range.
///
/// # Returns
///
/// The interpolated value.
pub fn cubic_spline_evaluate(x: &[f64], y: &[f64], moments: &[f64], widths: &[f64], t: f64) -> f64 {
    let n = x.len();
    let t = t.clamp(x[0], x[n - 1]);
    let i = segment_index(x, t);
    let width = widths[i];
    let ahead = x[i + 1] - t;
    let behind = t - x[i];

    (moments[i] / (6.0 * width)) * ahead * ahead * ahead
        + (moments[i + 1] / (6.0 * width)) * behind * behind * behind
        + (y[i] / width - moments[i] * width / 6.0) * ahead
        + (y[i + 1] / width - moments[i + 1] * width / 6.0) * behind
}

/// Returns the index of the segment containing `t`.
fn segment_index(x: &[f64], t: f64) -> usize {
    match x.binary_search_by(|value| value.total_cmp(&t)) {
        Ok(i) => i.min(x.len() - 2),
        Err(i) => i.saturating_sub(1).min(x.len() - 2),
    }
}

/// Returns the shape-preserving slope at an interior sample.
fn weighted_harmonic_slope(
    left_width: f64,
    right_width: f64,
    left_slope: f64,
    right_slope: f64,
) -> f64 {
    if left_slope * right_slope <= 0.0 {
        // A sign change is a local extremum; a zero slope keeps the interpolant monotone.
        return 0.0;
    }
    let w1 = 2.0 * right_width + left_width;
    let w2 = right_width + 2.0 * left_width;
    (w1 + w2) / (w1 / left_slope + w2 / right_slope)
}

/// Returns the one-sided slope at the first or last sample.
fn end_derivative(near_width: f64, far_width: f64, near_slope: f64, far_slope: f64) -> f64 {
    let total = near_width + far_width;
    if total.abs() < EPSILON {
        return near_slope;
    }

    let derivative = ((2.0 * near_width + far_width) * near_slope - near_width * far_slope) / total;
    if near_slope.abs() < EPSILON || derivative.signum() != near_slope.signum() {
        return 0.0;
    }
    if near_slope.signum() != far_slope.signum() && derivative.abs() > 3.0 * near_slope.abs() {
        return 3.0 * near_slope;
    }
    derivative
}

/// Returns the Akima slopes at every sample.
fn akima_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut secants: Vec<f64> = (0..n - 1)
        .map(|i| {
            let width = x[i + 1] - x[i];
            let rise = y[i + 1] - y[i];
            if width.abs() > EPSILON && rise.is_finite() {
                let slope = rise / width;
                if slope.is_finite() {
                    return slope;
                }
            }
            0.0
        })
        .collect();

    // Akima needs two secants beyond each end; they are extrapolated quadratically and
    // clamped, because an unbounded extrapolation would dominate the weighted average.
    extend_secants(&mut secants);

    (0..n)
        .map(|i| {
            let center = i + 2;
            let left_weight = (secants[center - 1] - secants[center - 2]).abs();
            let right_weight = (secants[center + 1] - secants[center]).abs();
            if left_weight + right_weight > EPSILON {
                (right_weight * secants[center - 1] + left_weight * secants[center])
                    / (right_weight + left_weight)
            } else {
                0.5 * (secants[center - 1] + secants[center])
            }
        })
        .collect()
}

/// Pads a secant series with two extrapolated values at each end.
fn extend_secants(secants: &mut Vec<f64>) {
    if secants.len() < 2 {
        let only = secants.first().copied().unwrap_or(0.0);
        secants.splice(0..0, [only, only]);
        secants.extend([only, only]);
        return;
    }

    let (front_far, front_near) = extrapolate(secants[0], secants[1]);
    secants.splice(0..0, [front_far, front_near]);

    let last = secants.len() - 1;
    let (back_far, back_near) = extrapolate(secants[last], secants[last - 1]);
    secants.extend([back_near, back_far]);
}

/// Extrapolates two secants beyond the end of a series, clamped to a sane magnitude.
fn extrapolate(nearest: f64, second: f64) -> (f64, f64) {
    let near = 2.0 * nearest - second;
    let far = 2.0 * near - nearest;
    let bound = nearest.abs().max(second.abs()) * 3.0 + EPSILON;
    let clamp = |value: f64| {
        if value.is_finite() && bound.is_finite() {
            value.clamp(-bound, bound)
        } else {
            0.0
        }
    };
    (clamp(far), clamp(near))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Abscissae at `t_i = (i + 1) / (n + 1)`, matching the resampling contract.
    fn abscissae(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i + 1) as f64 / (n + 1) as f64).collect()
    }

    #[test]
    fn pchip_reproduces_its_own_samples() {
        let x = abscissae(5);
        let y = vec![0.0, 2.0, 1.0, 4.0, 3.0];
        let derivatives = pchip_derivatives(&x, &y);

        for (xi, yi) in x.iter().zip(&y) {
            let value = pchip_evaluate(&x, &y, &derivatives, *xi);
            assert!((value - yi).abs() < 1e-9, "at {xi}: {value} vs {yi}");
        }
    }

    #[test]
    fn pchip_never_overshoots_a_monotone_series() {
        let x = abscissae(4);
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let derivatives = pchip_derivatives(&x, &y);

        for step in 0..=100 {
            let t = step as f64 / 100.0;
            let value = pchip_evaluate(&x, &y, &derivatives, t);
            assert!((0.0..=3.0).contains(&value), "at {t}: {value}");
        }
    }

    #[test]
    fn pchip_handles_degenerate_series() {
        assert_eq!(pchip_derivatives(&[], &[]), Vec::<f64>::new());
        assert_eq!(pchip_derivatives(&[0.5], &[7.0]), vec![0.0]);
        assert_eq!(pchip_evaluate(&[], &[], &[], 0.5), 0.0);
        assert_eq!(pchip_evaluate(&[0.5], &[7.0], &[0.0], 0.9), 7.0);
    }

    #[test]
    fn pchip_flattens_at_a_local_extremum() {
        let x = abscissae(3);
        let y = vec![0.0, 5.0, 0.0];

        let derivatives = pchip_derivatives(&x, &y);

        assert_eq!(derivatives[1], 0.0, "the peak must not overshoot");
    }

    #[test]
    fn akima_reproduces_its_own_samples() {
        let x = abscissae(6);
        let y = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let coefficients = akima_coefficients(&x, &y);

        for (xi, yi) in x.iter().zip(&y) {
            let value = akima_evaluate(&x, &y, &coefficients, *xi);
            assert!((value - yi).abs() < 1e-9, "at {xi}: {value} vs {yi}");
        }
    }

    #[test]
    fn akima_handles_degenerate_series() {
        assert!(akima_coefficients(&[0.5], &[1.0]).is_empty());
        assert_eq!(akima_evaluate(&[0.5], &[1.0], &[], 0.9), 1.0);
        assert_eq!(akima_evaluate(&[], &[], &[], 0.9), 0.0);
    }

    #[test]
    fn akima_stays_finite_on_a_constant_series() {
        let x = abscissae(5);
        let y = vec![2.0; 5];
        let coefficients = akima_coefficients(&x, &y);

        for step in 0..=20 {
            let value = akima_evaluate(&x, &y, &coefficients, step as f64 / 20.0);
            assert!((value - 2.0).abs() < 1e-9, "{value}");
        }
    }

    #[test]
    fn the_thomas_algorithm_solves_an_identity_system() {
        let solution =
            tridiagonal_thomas(&[1.0, 1.0, 1.0], &[0.0, 0.0], &[0.0, 0.0], &[4.0, 5.0, 6.0]);

        assert_eq!(solution, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn the_thomas_algorithm_solves_a_known_system() {
        // [2 1 0; 1 2 1; 0 1 2] x = [3, 4, 3]  =>  x = [1, 1, 1]
        let solution =
            tridiagonal_thomas(&[2.0, 2.0, 2.0], &[1.0, 1.0], &[1.0, 1.0], &[3.0, 4.0, 3.0]);

        for value in solution {
            assert!((value - 1.0).abs() < 1e-9, "{value}");
        }
    }

    #[test]
    fn a_singular_pivot_leaves_its_entry_at_zero() {
        let solution = tridiagonal_thomas(&[0.0, 1.0], &[0.0], &[0.0], &[5.0, 2.0]);

        assert_eq!(solution[0], 0.0, "no division by a vanishing pivot");
        assert_eq!(solution[1], 2.0);
    }

    #[test]
    fn the_segment_index_stays_in_range() {
        let x = abscissae(4);

        assert_eq!(segment_index(&x, -1.0), 0);
        assert_eq!(segment_index(&x, 2.0), x.len() - 2);
        assert_eq!(segment_index(&x, x[0]), 0);
    }
}
