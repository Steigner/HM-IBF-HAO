//! Dimension-aware boundary enforcement for heterogeneous islands.
//!
//! `boundary::Saturation` reads bounds from `problem.domain()`, which always returns
//! the max-dimension layout.  Islands running at a smaller IRACE-tuned dimension D
//! have control points at physically different backbone positions — each with its own
//! (lo, hi) pair from `sample_backbone_for_dimension(D)`.  Using max-dim bounds here
//! would either over-clamp (narrower true bounds) or under-clamp (wider true bounds),
//! producing solutions that violate the actual per-D geometry.
//!
//! `IslandDimensionSaturation` — post-mutation clamp for incremental operators (PSO,
//! DE, ES, LS, SA) that modify values in-place and may step outside bounds.
//!
//! `SafePartialRandomSpread` — replacement for `mutation::PartialRandomSpread` used by
//! RS.  The stock operator samples new values from `problem.domain()` (max-dim), whereas
//! RS islands may run at a smaller dimension D whose per-position bounds are different.
//! This version reads `IslandDimension` and samples from `domain_for_dimension(D)`.

use mahf::prelude::*;
use serde::{Deserialize, Serialize};

use crate::problems::{DimensionAwareDomain, RealValuedProblem};

/// Boundary saturation that clamps solutions to the bounds of the island's actual
/// working dimension instead of the problem's max-dimension `domain()`.
///
/// Reads `IslandDimension` from state (set during `RandomSpreadWithDimension::init`).
/// Falls back to `problem.dimension()` if the state entry is absent.
#[derive(Clone, Serialize, Deserialize)]
pub struct IslandDimensionSaturation;

impl IslandDimensionSaturation {
    /// Creates a new `IslandDimensionSaturation`.
    pub fn from_params() -> Self {
        Self
    }

    /// Wraps `IslandDimensionSaturation` in a boxed component.
    pub fn new<P: RealValuedProblem + DimensionAwareDomain>() -> Box<dyn Component<P>> {
        Box::new(Self::from_params())
    }
}

impl<P: RealValuedProblem + DimensionAwareDomain> Component<P> for IslandDimensionSaturation {
    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        use super::IslandDimension;

        let dim = state
            .try_get_value::<IslandDimension>()
            .unwrap_or_else(|_| problem.dimension());

        let bounds = problem.domain_for_dimension(dim);

        let mut populations = state.populations_mut();
        let population = populations.current_mut();

        for individual in population.iter_mut() {
            let solution = individual.solution_mut();
            // Only clamp positions that exist in bounds; extra elements (e.g. from a
            // migration that brought a longer solution) are left untouched — the evaluator
            // handles any out-of-bounds offset via its own clamp inside `evaluate_with`.
            for (val, range) in solution.iter_mut().zip(bounds.iter()) {
                *val = val.clamp(range.start, range.end);
            }
        }

        Ok(())
    }
}

// ============================================================================
// SafePartialRandomSpread
// ============================================================================

/// Dimension-aware replacement for `mutation::PartialRandomSpread`.
///
/// `PartialRandomSpread` samples new values from `problem.domain()`, which always
/// returns the max-dimension layout.  For RS islands running at dimension D < max_D
/// each control point sits at a different backbone position with its own (lo, hi)
/// from `sample_backbone_for_dimension(D)`.  This component reads `IslandDimension`
/// and samples from `domain_for_dimension(D)` so new random values stay within the
/// physically correct bounds for the active dimension.
///
/// The spread probability `p` controls how many dimensions are re-sampled per step:
/// each element is replaced with a fresh uniform draw with probability `p`.
#[derive(Clone, Serialize, Deserialize)]
pub struct SafePartialRandomSpread {
    /// Probability of re-sampling each dimension independently.
    pub p: f64,
}

impl SafePartialRandomSpread {
    /// Creates a new `SafePartialRandomSpread` with spread probability `p`.
    pub fn from_params(p: f64) -> Self {
        Self { p }
    }

    /// Wraps `SafePartialRandomSpread` in a boxed component.
    pub fn new<P: RealValuedProblem + DimensionAwareDomain>(p: f64) -> Box<dyn Component<P>> {
        Box::new(Self::from_params(p))
    }
}

impl<P: RealValuedProblem + DimensionAwareDomain> Component<P> for SafePartialRandomSpread {
    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        use super::IslandDimension;
        use mahf::rand::Rng;

        let dim = state
            .try_get_value::<IslandDimension>()
            .unwrap_or_else(|_| problem.dimension());

        let bounds = problem.domain_for_dimension(dim);

        // Phase 1: snapshot current solutions (releases the populations borrow).
        let current_solutions: Vec<Vec<f64>> = {
            let populations = state.populations();
            populations
                .current()
                .iter()
                .map(|ind| ind.solution().clone())
                .collect()
        };

        // Phase 2: generate new values using rng (no populations borrow held).
        let new_values: Vec<Vec<f64>> = {
            let mut rng = state.random_mut();
            current_solutions
                .iter()
                .map(|solution| {
                    solution
                        .iter()
                        .zip(bounds.iter())
                        .map(|(val, range)| {
                            if rng.gen::<f64>() < self.p {
                                rng.gen_range(range.clone())
                            } else {
                                *val
                            }
                        })
                        .collect()
                })
                .collect()
        };

        // Phase 3: write back.
        let mut populations = state.populations_mut();
        let population = populations.current_mut();
        for (individual, new_solution) in population.iter_mut().zip(new_values) {
            *individual.solution_mut() = new_solution;
        }

        Ok(())
    }
}
