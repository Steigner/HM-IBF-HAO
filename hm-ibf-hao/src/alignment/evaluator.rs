//! The dimension-agnostic objective evaluator.

use mahf::{problems::Evaluate, Individual, State};

use super::problem::HorizontalAlignment;

/// Scores horizontal alignment solutions of any working dimension.
///
/// The solution's own length selects how many control points are placed on the backbone, so
/// one evaluator serves every island regardless of the dimension IRACE assigned it.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlignmentEvaluator;

impl Evaluate for AlignmentEvaluator {
    type Problem = HorizontalAlignment;

    fn evaluate(
        &mut self,
        problem: &Self::Problem,
        _state: &mut State<Self::Problem>,
        individuals: &mut [Individual<Self::Problem>],
    ) {
        for individual in individuals {
            individual.evaluate_with(|offsets: &Vec<f64>| {
                problem
                    .evaluate_solution(offsets)
                    .try_into()
                    .expect("the objective is finite and non-negative by construction")
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use mahf::{population::IntoIndividuals, problems::KnownOptimumProblem, Random};

    use super::{
        super::test_support::{flat_instance, FIXTURE_DIMENSIONS},
        *,
    };

    /// Evaluates a batch of solutions through the full `mahf` evaluator interface.
    fn evaluate(
        instance: &HorizontalAlignment,
        solutions: Vec<Vec<f64>>,
    ) -> Vec<Individual<HorizontalAlignment>> {
        let mut state = State::new();
        state.insert(Random::new(0));
        let mut individuals = solutions.into_individuals();

        AlignmentEvaluator.evaluate(instance, &mut state, &mut individuals);

        individuals
    }

    #[test]
    fn every_individual_of_a_batch_is_scored() {
        let instance = flat_instance();

        let individuals = evaluate(&instance, vec![vec![0.0; 3], vec![1.0; 3], vec![-1.0; 3]]);

        assert_eq!(individuals.len(), 3);
        assert!(individuals.iter().all(|i| i.get_objective().is_some()));
    }

    #[test]
    fn the_evaluator_agrees_with_the_problem() {
        let instance = flat_instance();
        let offsets = vec![0.5, -0.5, 1.5];

        let individuals = evaluate(&instance, vec![offsets.clone()]);

        assert_eq!(
            individuals[0].objective().value(),
            instance.evaluate_solution(&offsets)
        );
    }

    #[test]
    fn solutions_of_mixed_dimensions_are_scored_in_one_batch() {
        let instance = flat_instance();
        let solutions: Vec<Vec<f64>> = FIXTURE_DIMENSIONS
            .iter()
            .map(|&dimension| vec![0.0; dimension as usize])
            .collect();

        let individuals = evaluate(&instance, solutions);

        assert_eq!(individuals.len(), FIXTURE_DIMENSIONS.len());
        for individual in &individuals {
            let value = individual.objective().value();
            assert!(value.is_finite() && value >= 0.0, "got {value}");
        }
    }

    #[test]
    fn an_empty_batch_is_accepted() {
        let instance = flat_instance();

        assert!(evaluate(&instance, Vec::new()).is_empty());
    }

    #[test]
    fn the_known_optimum_is_the_lower_bound_of_the_objective() {
        let instance = flat_instance();

        let individuals = evaluate(&instance, vec![vec![0.0; 5]]);

        assert!(individuals[0].objective() >= &instance.known_optimum());
    }
}
