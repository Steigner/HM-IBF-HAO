//! Arc-length resampling of an offset vector between two working dimensions.
//!
//! Every method treats the offsets as a plain 1-D signal sampled at
//! `t_i = (i + 1) / (D + 1)`, which is exactly where
//! [`HorizontalAlignment::sample_backbone`](crate::alignment::HorizontalAlignment::sample_backbone)
//! places its control points. Sampling at `i / (D - 1)` instead would pin the first and last
//! offset to the backbone's endpoints, which carry no offset at all, and shift every value in
//! between.

use super::interpolation::{
    akima_coefficients, akima_evaluate, cubic_spline_evaluate, pchip_derivatives, pchip_evaluate,
    tridiagonal_thomas,
};

/// Denoising weight of the total-variation method.
const TV_LAMBDA: f64 = 1.0;

/// Augmented-Lagrangian penalty of the total-variation method.
const TV_RHO: f64 = 1.0;

/// Iteration count of the total-variation method's ADMM loop.
const TV_ITERATIONS: usize = 100;

/// Returns the abscissae a solution of the given dimension is sampled at.
///
/// # Arguments
///
/// * `dimension` - The working dimension.
///
/// # Returns
///
/// The abscissae `t_i = (i + 1) / (dimension + 1)`.
pub fn abscissae(dimension: usize) -> Vec<f64> {
    (0..dimension)
        .map(|i| (i + 1) as f64 / (dimension + 1) as f64)
        .collect()
}

/// Resamples by piecewise-linear interpolation.
///
/// # Arguments
///
/// * `solution` - The source offsets.
/// * `target_dim` - The required length.
///
/// # Returns
///
/// The resampled offsets.
pub fn arc_linear(solution: &[f64], target_dim: usize) -> Vec<f64> {
    let n = solution.len();
    if let Some(constant) = degenerate(solution, target_dim) {
        return constant;
    }

    abscissae(target_dim)
        .into_iter()
        .map(|t| {
            // Invert t = (i + 1) / (n + 1) to find the fractional source index.
            let index = t * (n + 1) as f64 - 1.0;
            let low = (index.floor().max(0.0) as usize).min(n - 1);
            let high = (low + 1).min(n - 1);
            let fraction = (index - low as f64).clamp(0.0, 1.0);
            solution[low] * (1.0 - fraction) + solution[high] * fraction
        })
        .collect()
}

/// Resamples with a shape-preserving PCHIP interpolant.
///
/// # Arguments
///
/// * `solution` - The source offsets.
/// * `target_dim` - The required length.
///
/// # Returns
///
/// The resampled offsets.
pub fn arc_pchip(solution: &[f64], target_dim: usize) -> Vec<f64> {
    if let Some(constant) = degenerate(solution, target_dim) {
        return constant;
    }

    let source = abscissae(solution.len());
    let derivatives = pchip_derivatives(&source, solution);

    abscissae(target_dim)
        .into_iter()
        .map(|t| pchip_evaluate(&source, solution, &derivatives, t))
        .collect()
}

/// Resamples with a locally adaptive Akima spline.
///
/// # Arguments
///
/// * `solution` - The source offsets.
/// * `target_dim` - The required length.
///
/// # Returns
///
/// The resampled offsets.
pub fn arc_akima(solution: &[f64], target_dim: usize) -> Vec<f64> {
    if let Some(constant) = degenerate(solution, target_dim) {
        return constant;
    }

    let source = abscissae(solution.len());
    let coefficients = akima_coefficients(&source, solution);

    abscissae(target_dim)
        .into_iter()
        .map(|t| akima_evaluate(&source, solution, &coefficients, t))
        .collect()
}

/// Resamples with a clamped cubic spline.
///
/// The boundary conditions hold the first derivative at zero, which keeps the corridor from
/// leaving the backbone at an angle where it is pinned to it.
///
/// # Arguments
///
/// * `solution` - The source offsets.
/// * `target_dim` - The required length.
///
/// # Returns
///
/// The resampled offsets.
pub fn arc_clamped_cubic(solution: &[f64], target_dim: usize) -> Vec<f64> {
    if let Some(constant) = degenerate(solution, target_dim) {
        return constant;
    }
    let n = solution.len();
    if n == 2 {
        return arc_linear(solution, target_dim);
    }

    let source = abscissae(n);
    let widths: Vec<f64> = (0..n - 1).map(|i| source[i + 1] - source[i]).collect();
    let secants: Vec<f64> = (0..n - 1)
        .map(|i| (solution[i + 1] - solution[i]) / widths[i])
        .collect();

    let mut diagonal = vec![0.0; n];
    let mut upper = vec![0.0; n - 1];
    let mut lower = vec![0.0; n - 1];
    let mut rhs = vec![0.0; n];

    diagonal[0] = 2.0 * widths[0];
    upper[0] = widths[0];
    rhs[0] = 6.0 * secants[0];

    for i in 1..n - 1 {
        lower[i - 1] = widths[i - 1];
        diagonal[i] = 2.0 * (widths[i - 1] + widths[i]);
        upper[i] = widths[i];
        rhs[i] = 6.0 * (secants[i] - secants[i - 1]);
    }

    lower[n - 2] = widths[n - 2];
    diagonal[n - 1] = 2.0 * widths[n - 2];
    rhs[n - 1] = -6.0 * secants[n - 2];

    let moments = tridiagonal_thomas(&diagonal, &upper, &lower, &rhs);

    abscissae(target_dim)
        .into_iter()
        .map(|t| cubic_spline_evaluate(&source, solution, &moments, &widths, t))
        .collect()
}

/// Resamples after edge-preserving total-variation denoising.
///
/// Solves `min_u ½‖u − y‖² + λ‖Du‖₁` with ADMM, then resamples the denoised signal with
/// PCHIP. Unlike the plain spline methods this keeps sharp offset changes sharp instead of
/// rounding them away.
///
/// # Arguments
///
/// * `solution` - The source offsets.
/// * `target_dim` - The required length.
///
/// # Returns
///
/// The resampled offsets.
pub fn arc_total_variation(solution: &[f64], target_dim: usize) -> Vec<f64> {
    if let Some(constant) = degenerate(solution, target_dim) {
        return constant;
    }
    let n = solution.len();
    if n == 2 {
        return arc_linear(solution, target_dim);
    }

    // (I + rho·DᵀD) is tridiagonal with main diagonal [1+rho, 1+2rho, .., 1+2rho, 1+rho].
    let mut diagonal = vec![1.0 + 2.0 * TV_RHO; n];
    diagonal[0] = 1.0 + TV_RHO;
    diagonal[n - 1] = 1.0 + TV_RHO;
    let off_diagonal = vec![-TV_RHO; n - 1];

    let mut denoised = solution.to_vec();
    let mut split = vec![0.0; n - 1];
    let mut dual = vec![0.0; n - 1];

    for _ in 0..TV_ITERATIONS {
        let mut transposed = vec![0.0; n];
        for i in 0..n - 1 {
            let value = split[i] - dual[i];
            transposed[i] -= value;
            transposed[i + 1] += value;
        }
        let rhs: Vec<f64> = (0..n)
            .map(|i| solution[i] + TV_RHO * transposed[i])
            .collect();

        denoised = tridiagonal_thomas(&diagonal, &off_diagonal, &off_diagonal, &rhs);

        for i in 0..n - 1 {
            let value = (denoised[i + 1] - denoised[i]) + dual[i];
            split[i] = value.signum() * (value.abs() - TV_LAMBDA / TV_RHO).max(0.0);
            dual[i] += (denoised[i + 1] - denoised[i]) - split[i];
        }
    }

    let source = abscissae(n);
    let derivatives = pchip_derivatives(&source, &denoised);

    abscissae(target_dim)
        .into_iter()
        .map(|t| pchip_evaluate(&source, &denoised, &derivatives, t))
        .collect()
}

/// Returns the resampling of a source too short to interpolate, if it is one.
///
/// # Arguments
///
/// * `solution` - The source offsets.
/// * `target_dim` - The required length.
///
/// # Returns
///
/// A constant vector of length `target_dim`, or `None` when the source has two or more
/// offsets and can be interpolated normally.
fn degenerate(solution: &[f64], target_dim: usize) -> Option<Vec<f64>> {
    match solution.len() {
        0 => Some(vec![0.0; target_dim]),
        1 => Some(vec![solution[0]; target_dim]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A resampling method's signature.
    type Method = fn(&[f64], usize) -> Vec<f64>;

    /// Every resampling method, by name.
    const METHODS: [(&str, Method); 5] = [
        ("linear", arc_linear),
        ("pchip", arc_pchip),
        ("akima", arc_akima),
        ("clamped_cubic", arc_clamped_cubic),
        ("total_variation", arc_total_variation),
    ];

    #[test]
    fn the_abscissae_are_interior_and_ascending() {
        let x = abscissae(4);

        assert_eq!(x, vec![0.2, 0.4, 0.6, 0.8]);
        assert!(x.iter().all(|&t| 0.0 < t && t < 1.0));
    }

    #[test]
    fn every_method_produces_the_requested_length() {
        let source = vec![1.0, -2.0, 3.0, 0.5, -1.5];

        for (name, method) in METHODS {
            for target in [1, 2, 5, 13, 79] {
                let output = method(&source, target);
                assert_eq!(output.len(), target, "{name} to {target}");
            }
        }
    }

    #[test]
    fn every_method_produces_finite_values() {
        let source = vec![1e6, -1e6, 0.0, 42.0];

        for (name, method) in METHODS {
            let output = method(&source, 37);
            assert!(output.iter().all(|v| v.is_finite()), "{name}: {output:?}");
        }
    }

    #[test]
    fn every_method_preserves_a_constant_signal() {
        let source = vec![2.5; 6];

        for (name, method) in METHODS {
            let output = method(&source, 11);
            assert!(
                output.iter().all(|v| (v - 2.5).abs() < 1e-6),
                "{name}: {output:?}"
            );
        }
    }

    #[test]
    fn every_method_handles_an_empty_source() {
        for (name, method) in METHODS {
            assert_eq!(method(&[], 4), vec![0.0; 4], "{name}");
        }
    }

    #[test]
    fn every_method_broadcasts_a_single_offset() {
        for (name, method) in METHODS {
            assert_eq!(method(&[3.5], 4), vec![3.5; 4], "{name}");
        }
    }

    #[test]
    fn every_method_handles_a_two_element_source() {
        for (name, method) in METHODS {
            let output = method(&[0.0, 1.0], 7);
            assert_eq!(output.len(), 7, "{name}");
            assert!(output.iter().all(|v| v.is_finite()), "{name}: {output:?}");
        }
    }

    #[test]
    fn linear_resampling_to_the_same_dimension_is_the_identity() {
        let source = vec![1.0, -2.0, 3.0, 0.5];

        let output = arc_linear(&source, source.len());

        for (before, after) in source.iter().zip(&output) {
            assert!((before - after).abs() < 1e-9, "{before} became {after}");
        }
    }

    #[test]
    fn pchip_stays_inside_the_range_of_its_source() {
        let source = vec![0.0, 10.0, -5.0, 7.0];
        let (low, high) = (-5.0, 10.0);

        let output = arc_pchip(&source, 40);

        assert!(
            output.iter().all(|&v| (low..=high).contains(&v)),
            "{output:?}"
        );
    }

    #[test]
    fn total_variation_flattens_a_noisy_plateau() {
        let noisy: Vec<f64> = (0..12)
            .map(|i| if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect();

        let output = arc_total_variation(&noisy, 12);

        let spread = output.iter().cloned().fold(f64::MIN, f64::max)
            - output.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            spread < 0.1,
            "denoising must shrink the spread, got {spread}"
        );
    }

    #[test]
    fn total_variation_keeps_a_step_edge() {
        let step: Vec<f64> = (0..12).map(|i| if i < 6 { 0.0 } else { 10.0 }).collect();

        let output = arc_total_variation(&step, 12);

        assert!(output[0] < 2.0, "the low plateau must survive: {output:?}");
        assert!(
            output[11] > 8.0,
            "the high plateau must survive: {output:?}"
        );
    }
}
