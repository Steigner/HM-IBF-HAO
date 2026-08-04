//! Resizing of solutions that migrate between islands of different working dimensions.
//!
//! A migrant is an offset vector sampled along the backbone. Moving it to an island of a
//! different dimension means resampling that signal at the target island's control points;
//! IRACE picks which of the [`TransformMethod`] variants each migration edge uses.

pub mod interpolation;
pub mod resample;

use grahf::components::transform::{SolutionTransformer, TransformRequest};
use mahf::{rand, Random};
use serde::{Deserialize, Serialize};

use crate::problems::{DimensionAwareDomain, RealValuedProblem};

/// The resampling methods IRACE may assign to a migration edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformMethod {
    /// Piecewise-linear resampling; the cheapest deterministic option.
    ArcLinear,
    /// Shape-preserving PCHIP; never overshoots the source's range.
    ArcPchip,
    /// Akima spline; locally adaptive and stable across sharp terrain changes.
    ArcAkima,
    /// Clamped cubic spline; smoothest, with zero-derivative boundary conditions.
    ArcClampedCubic,
    /// Total-variation denoising followed by PCHIP; edge preserving.
    ArcTotalVariation,
}

impl TransformMethod {
    /// Returns every method name exposed to IRACE.
    ///
    /// # Returns
    ///
    /// The names, in the order the variants are declared.
    pub fn all_names() -> Vec<&'static str> {
        vec![
            "arc_linear",
            "arc_pchip",
            "arc_akima",
            "arc_clamped_cubic",
            "arc_total_variation",
        ]
    }

    /// Parses a method from its IRACE name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name as it appears in the parameter space.
    ///
    /// # Returns
    ///
    /// The method, or `None` if the name is unknown.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "arc_linear" => Some(Self::ArcLinear),
            "arc_pchip" => Some(Self::ArcPchip),
            "arc_akima" => Some(Self::ArcAkima),
            "arc_clamped_cubic" => Some(Self::ArcClampedCubic),
            "arc_total_variation" => Some(Self::ArcTotalVariation),
            _ => None,
        }
    }

    /// Applies the method.
    ///
    /// # Arguments
    ///
    /// * `solution` - The source offsets.
    /// * `target_dim` - The required length.
    ///
    /// # Returns
    ///
    /// The resampled offsets.
    pub fn apply(self, solution: &[f64], target_dim: usize) -> Vec<f64> {
        match self {
            Self::ArcLinear => resample::arc_linear(solution, target_dim),
            Self::ArcPchip => resample::arc_pchip(solution, target_dim),
            Self::ArcAkima => resample::arc_akima(solution, target_dim),
            Self::ArcClampedCubic => resample::arc_clamped_cubic(solution, target_dim),
            Self::ArcTotalVariation => resample::arc_total_variation(solution, target_dim),
        }
    }
}

/// Resamples an offset vector to a target dimension by method name.
///
/// An unknown method name falls back to [`TransformMethod::ArcLinear`]: the name comes from
/// a stored IRACE parameter file, and refusing to migrate would silently change the
/// algorithm being replayed.
///
/// # Arguments
///
/// * `solution` - The source offsets.
/// * `source_dim` - The dimension the offsets were sampled at; must match their length.
/// * `target_dim` - The required length.
/// * `method_name` - The IRACE name of the resampling method.
/// * `rng` - Unused; present because the transform interface allows stochastic methods.
///
/// # Returns
///
/// The resampled offsets, of length `target_dim`.
pub fn transform_with_optional_params(
    solution: &[f64],
    source_dim: usize,
    target_dim: usize,
    method_name: &str,
    rng: &mut impl rand::Rng,
) -> Vec<f64> {
    let _ = rng;
    debug_assert_eq!(
        source_dim,
        solution.len(),
        "the request's source dimension must match the solution's length"
    );

    if solution.len() == target_dim {
        return solution.to_vec();
    }

    TransformMethod::from_name(method_name)
        .unwrap_or(TransformMethod::ArcLinear)
        .apply(solution, target_dim)
}

/// Resizes migrants by resampling their offsets along the backbone.
///
/// After resampling, the offsets are clamped into the target dimension's own bounds: those
/// bounds sit at different backbone positions for every dimension, so a migrant that was
/// admissible at its source dimension can otherwise arrive outside the terrain.
#[derive(Clone, Copy, Debug, Default)]
pub struct OffsetResampleTransformer;

impl OffsetResampleTransformer {
    /// Creates the transformer.
    ///
    /// # Returns
    ///
    /// The transformer value.
    pub fn new() -> Self {
        Self
    }
}

impl<P> SolutionTransformer<P> for OffsetResampleTransformer
where
    P: RealValuedProblem + DimensionAwareDomain + Send + Sync + 'static,
{
    fn transform(
        &self,
        problem: &P,
        solution: &P::Encoding,
        request: TransformRequest<'_>,
        rng: &mut Random,
    ) -> P::Encoding {
        let mut resized = transform_with_optional_params(
            solution,
            solution.len(),
            request.target_dim as usize,
            request.method,
            rng,
        );

        for (offset, bounds) in resized
            .iter_mut()
            .zip(problem.domain_for_dimension(request.target_dim as usize))
        {
            *offset = offset.clamp(bounds.start, bounds.end);
        }

        resized
    }
}

#[cfg(test)]
mod tests {
    use mahf::problems::{LimitedVectorProblem, VectorProblem};

    use super::*;
    use crate::alignment::test_support::{flat_instance, FIXTURE_DIMENSIONS};

    #[test]
    fn every_name_round_trips_through_the_enum() {
        for name in TransformMethod::all_names() {
            let method = TransformMethod::from_name(name).expect(name);
            assert_eq!(
                TransformMethod::all_names()
                    .iter()
                    .position(|candidate| *candidate == name),
                TransformMethod::all_names()
                    .iter()
                    .position(|candidate| TransformMethod::from_name(candidate) == Some(method)),
            );
        }
    }

    #[test]
    fn an_unknown_name_is_not_parsed() {
        assert!(TransformMethod::from_name("bicubic").is_none());
    }

    #[test]
    fn an_unknown_name_falls_back_to_linear() {
        let mut rng = Random::new(0);
        let source = vec![1.0, 2.0, 3.0];

        let fallback = transform_with_optional_params(&source, 3, 7, "bicubic", &mut rng);
        let linear = resample::arc_linear(&source, 7);

        assert_eq!(fallback, linear);
    }

    #[test]
    fn resampling_to_the_same_dimension_returns_the_source() {
        let mut rng = Random::new(0);
        let source = vec![1.0, 2.0, 3.0];

        for name in TransformMethod::all_names() {
            assert_eq!(
                transform_with_optional_params(&source, 3, 3, name, &mut rng),
                source,
                "{name}"
            );
        }
    }

    #[test]
    fn migration_between_any_two_allowed_dimensions_yields_a_valid_solution() {
        let instance = flat_instance();
        let transformer = OffsetResampleTransformer::new();
        let mut rng = Random::new(0);

        for &source_dim in &FIXTURE_DIMENSIONS {
            for &target_dim in &FIXTURE_DIMENSIONS {
                for method in TransformMethod::all_names() {
                    let input: Vec<f64> = (0..source_dim)
                        .map(|i| (i as f64 * 0.37).sin() * 3.0)
                        .collect();
                    let request = TransformRequest::new(source_dim, target_dim, method);

                    let output = transformer.transform(&instance, &input, request, &mut rng);

                    assert_eq!(
                        output.len(),
                        target_dim as usize,
                        "{method}: {source_dim} -> {target_dim}"
                    );
                    assert!(
                        output.iter().all(|v| v.is_finite()),
                        "{method}: {source_dim} -> {target_dim} produced {output:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn migrants_are_clamped_into_the_target_dimension_bounds() {
        let instance = flat_instance();
        let transformer = OffsetResampleTransformer::new();
        let mut rng = Random::new(1);
        let target_dim = FIXTURE_DIMENSIONS[2];

        for method in TransformMethod::all_names() {
            let input = vec![1e9; FIXTURE_DIMENSIONS[0] as usize];
            let request = TransformRequest::new(FIXTURE_DIMENSIONS[0], target_dim, method);

            let output = transformer.transform(&instance, &input, request, &mut rng);

            for (offset, bounds) in output
                .iter()
                .zip(instance.domain_for_dimension(target_dim as usize))
            {
                assert!(
                    (bounds.start..=bounds.end).contains(offset),
                    "{method} produced {offset} outside [{}, {}]",
                    bounds.start,
                    bounds.end
                );
            }
        }
    }

    #[test]
    fn a_migrant_stays_evaluable_after_resizing() {
        let instance = flat_instance();
        let transformer = OffsetResampleTransformer::new();
        let mut rng = Random::new(2);
        let input: Vec<f64> = (0..FIXTURE_DIMENSIONS[0]).map(|i| i as f64 * 0.1).collect();

        let output = transformer.transform(
            &instance,
            &input,
            TransformRequest::new(FIXTURE_DIMENSIONS[0], FIXTURE_DIMENSIONS[2], "arc_pchip"),
            &mut rng,
        );

        assert!(instance.evaluate_solution(&output).is_finite());
        assert_eq!(instance.domain().len(), instance.dimension());
    }
}
