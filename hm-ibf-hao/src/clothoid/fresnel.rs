//! Closed-form approximation of the Fresnel integrals.
//!
//! A clothoid is the curve whose Cartesian coordinates are the Fresnel integrals `C` and
//! `S`. Evaluating them by quadrature inside the objective function would dominate its
//! runtime, so the rational approximation of McCrae and Singh, "Sketching piecewise
//! clothoid curves" (SBIM 2008) is used instead. Its absolute error stays below `2e-3`
//! over the whole domain, which is well below the pixel resolution of the heightmaps.

use std::f64::consts::{PI, SQRT_2};

/// Evaluates the Fresnel cosine integral `C(phi)`.
///
/// # Arguments
///
/// * `phi` - The normalized arc-length parameter.
///
/// # Returns
///
/// The approximated value of `C(phi)`.
#[inline]
pub fn cosine_integral(phi: f64) -> f64 {
    0.5 - radius(phi) * (0.5 * PI * (amplitude(phi) - phi * phi)).sin()
}

/// Evaluates the Fresnel sine integral `S(phi)`.
///
/// # Arguments
///
/// * `phi` - The normalized arc-length parameter.
///
/// # Returns
///
/// The approximated value of `S(phi)`.
#[inline]
pub fn sine_integral(phi: f64) -> f64 {
    0.5 - radius(phi) * (0.5 * PI * (amplitude(phi) - phi * phi)).cos()
}

/// Evaluates both Fresnel integrals at once.
///
/// # Arguments
///
/// * `phi` - The normalized arc-length parameter.
///
/// # Returns
///
/// The pair `(C(phi), S(phi))`.
#[inline]
pub fn integrals(phi: f64) -> (f64, f64) {
    (cosine_integral(phi), sine_integral(phi))
}

/// Evaluates the `R(phi)` envelope term of the approximation.
#[inline]
fn radius(phi: f64) -> f64 {
    (0.506 * phi + 1.0) / (1.79 * phi * phi + 2.054 * phi + SQRT_2)
}

/// Evaluates the `A(phi)` phase term of the approximation.
#[inline]
fn amplitude(phi: f64) -> f64 {
    1.0 / (0.803 * phi * phi * phi + 1.886 * phi * phi + 2.524 * phi + 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values obtained by Simpson quadrature of the defining integrals.
    ///
    /// `C(phi) = ∫₀^phi cos(π t²/2) dt`, `S(phi) = ∫₀^phi sin(π t²/2) dt`.
    fn reference(phi: f64) -> (f64, f64) {
        const STEPS: usize = 200_000;
        let h = phi / STEPS as f64;
        let (mut c, mut s) = (0.0, 0.0);
        for i in 0..STEPS {
            let (t0, t1) = (i as f64 * h, (i as f64 + 0.5) * h);
            let t2 = (i as f64 + 1.0) * h;
            let f = |t: f64| (0.5 * PI * t * t).cos();
            let g = |t: f64| (0.5 * PI * t * t).sin();
            c += h / 6.0 * (f(t0) + 4.0 * f(t1) + f(t2));
            s += h / 6.0 * (g(t0) + 4.0 * g(t1) + g(t2));
        }
        (c, s)
    }

    #[test]
    fn both_integrals_vanish_at_the_origin() {
        let (c, s) = integrals(0.0);

        assert!(c.abs() < 1e-3, "C(0) = {c}");
        assert!(s.abs() < 1e-3, "S(0) = {s}");
    }

    #[test]
    fn the_approximation_tracks_the_quadrature_reference() {
        for step in 1..=20 {
            let phi = step as f64 * 0.1;
            let (c, s) = integrals(phi);
            let (c_ref, s_ref) = reference(phi);

            assert!((c - c_ref).abs() < 2e-3, "C({phi}): {c} vs {c_ref}");
            assert!((s - s_ref).abs() < 2e-3, "S({phi}): {s} vs {s_ref}");
        }
    }

    #[test]
    fn the_integrals_converge_to_one_half_for_large_arguments() {
        let (c, s) = integrals(50.0);

        assert!((c - 0.5).abs() < 1e-2, "C(50) = {c}");
        assert!((s - 0.5).abs() < 1e-2, "S(50) = {s}");
    }

    #[test]
    fn the_split_and_combined_entry_points_agree() {
        let (c, s) = integrals(0.7);

        assert_eq!(c, cosine_integral(0.7));
        assert_eq!(s, sine_integral(0.7));
    }
}
