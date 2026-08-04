//! Dimension-aware diversity component that uses `solution.len()` instead of
//! `problem.dimension()` to avoid out-of-bounds access when IRACE assigns a custom dimension.

use better_any::{Tid, TidAble};
use mahf::prelude::*;
use mahf::CustomState;
use serde::{Deserialize, Serialize};

use crate::problems::RealValuedProblem;

/// Custom diversity state that stores the computed value.
#[derive(Tid, Clone, Default)]
pub struct SafeDiversityValue(pub f64);

impl CustomState<'_> for SafeDiversityValue {}

/// Safe Pairwise Distance Diversity that uses solution.len() instead of problem.dimension().
#[derive(Clone, Serialize, Deserialize)]
pub struct SafePairwiseDistanceDiversity;

impl SafePairwiseDistanceDiversity {
    /// Creates a new `SafePairwiseDistanceDiversity`.
    pub fn from_params() -> Self {
        Self
    }

    /// Wraps `SafePairwiseDistanceDiversity` in a boxed component.
    pub fn new<P: RealValuedProblem>() -> Box<dyn Component<P>> {
        Box::new(Self::from_params())
    }
}

impl<P: RealValuedProblem> Component<P> for SafePairwiseDistanceDiversity {
    fn execute(&self, _problem: &P, state: &mut State<P>) -> ExecResult<()> {
        let populations = state.populations();
        let population = populations.current();

        if population.is_empty() {
            return Ok(());
        }

        let n = population.len();
        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let sol_i = population[i].solution();
                let sol_j = population[j].solution();

                // Use minimum of the two solution lengths to avoid index out of bounds
                let dim = sol_i.len().min(sol_j.len());

                if dim > 0 {
                    let distance: f64 = (0..dim)
                        .map(|k| {
                            let diff = sol_i[k] - sol_j[k];
                            diff * diff
                        })
                        .sum::<f64>()
                        .sqrt();

                    total_distance += distance;
                    count += 1;
                }
            }
        }

        drop(populations);

        let avg_diversity = if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        };

        state.insert(SafeDiversityValue(avg_diversity));

        Ok(())
    }
}
