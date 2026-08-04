//! Dimension-aware DE crossover components that use `solution.len()` instead of
//! `problem.dimension()` to avoid out-of-bounds access when IRACE assigns a custom dimension.

use mahf::population::IntoIndividuals;
use mahf::prelude::*;
use serde::{Deserialize, Serialize};

use crate::problems::RealValuedProblem;

/// Safe DE Binomial Crossover that uses solution.len() instead of problem.dimension().
#[derive(Clone, Serialize, Deserialize)]
pub struct SafeDEBinomialCrossover {
    /// Crossover probability.
    pub pc: f64,
}

impl SafeDEBinomialCrossover {
    /// Creates a new `SafeDEBinomialCrossover` with the given crossover probability.
    pub fn from_params(pc: f64) -> Self {
        Self { pc }
    }

    /// Wraps `SafeDEBinomialCrossover` in a boxed component.
    pub fn new<P: RealValuedProblem>(pc: f64) -> Box<dyn Component<P>> {
        Box::new(Self::from_params(pc))
    }
}

impl<P: RealValuedProblem> Component<P> for SafeDEBinomialCrossover {
    fn execute(&self, _problem: &P, state: &mut State<P>) -> ExecResult<()> {
        use mahf::rand::Rng;

        // Pop offspring (mutants) from stack
        let mut populations = state.populations_mut();
        let offspring = populations.pop();
        let parents = populations.current();

        // Collect data
        let parents_data: Vec<Vec<f64>> = parents.iter().map(|p| p.solution().clone()).collect();
        let children_data: Vec<Vec<f64>> = offspring.iter().map(|c| c.solution().clone()).collect();

        drop(populations);

        // Compute crossover results
        let len = parents_data.len().min(children_data.len());
        let mut new_solutions: Vec<Vec<f64>> = Vec::with_capacity(len);

        {
            let mut rng = state.random_mut();

            for idx in 0..len {
                let parent_solution: &Vec<f64> = &parents_data[idx];
                let child_solution: &Vec<f64> = &children_data[idx];

                // Use actual solution length, not problem.dimension()
                let dim = child_solution.len().min(parent_solution.len());

                if dim == 0 {
                    new_solutions.push(child_solution.clone());
                    continue;
                }

                // Random index that will definitely be crossed
                let j_rand = rng.gen_range(0..dim);

                // Create new solution with crossover
                let mut new_solution = child_solution.clone();
                for i in 0..dim {
                    if i == j_rand || rng.gen::<f64>() < self.pc {
                        // Keep mutant value (already in child)
                    } else {
                        // Use parent value
                        new_solution[i] = parent_solution[i];
                    }
                }

                new_solutions.push(new_solution);
            }
        }

        // Create new offspring population with crossover results
        let new_offspring: Vec<_> = new_solutions.into_individuals();

        // Push the new offspring back
        state.populations_mut().push(new_offspring);

        Ok(())
    }
}

/// Safe DE Exponential Crossover that uses solution.len() instead of problem.dimension().
#[derive(Clone, Serialize, Deserialize)]
pub struct SafeDEExponentialCrossover {
    /// Crossover probability.
    pub pc: f64,
}

impl SafeDEExponentialCrossover {
    /// Creates a new `SafeDEExponentialCrossover` with the given crossover probability.
    pub fn from_params(pc: f64) -> Self {
        Self { pc }
    }

    /// Wraps `SafeDEExponentialCrossover` in a boxed component.
    pub fn new<P: RealValuedProblem>(pc: f64) -> Box<dyn Component<P>> {
        Box::new(Self::from_params(pc))
    }
}

impl<P: RealValuedProblem> Component<P> for SafeDEExponentialCrossover {
    fn execute(&self, _problem: &P, state: &mut State<P>) -> ExecResult<()> {
        use mahf::rand::Rng;

        // Pop offspring (mutants) from stack
        let mut populations = state.populations_mut();
        let offspring = populations.pop();
        let parents = populations.current();

        // Collect data
        let parents_data: Vec<Vec<f64>> = parents.iter().map(|p| p.solution().clone()).collect();
        let children_data: Vec<Vec<f64>> = offspring.iter().map(|c| c.solution().clone()).collect();

        drop(populations);

        let len = parents_data.len().min(children_data.len());
        let mut new_solutions: Vec<Vec<f64>> = Vec::with_capacity(len);

        {
            let mut rng = state.random_mut();

            for idx in 0..len {
                let parent_solution: &Vec<f64> = &parents_data[idx];
                let child_solution: &Vec<f64> = &children_data[idx];

                // Use actual solution length, not problem.dimension()
                let dim = child_solution.len().min(parent_solution.len());

                if dim == 0 {
                    new_solutions.push(child_solution.clone());
                    continue;
                }

                // Random starting point
                let start = rng.gen_range(0..dim);

                // Create new solution starting from parent
                let mut new_solution = parent_solution.clone();

                // Copy from mutant starting at random position, continuing while pc succeeds
                let mut i = start;
                loop {
                    new_solution[i] = child_solution[i];
                    i = (i + 1) % dim;
                    if i == start || rng.gen::<f64>() >= self.pc {
                        break;
                    }
                }

                new_solutions.push(new_solution);
            }
        }

        // Create new offspring population with crossover results
        let new_offspring: Vec<_> = new_solutions.into_individuals();

        // Push the new offspring back
        state.populations_mut().push(new_offspring);

        Ok(())
    }
}
