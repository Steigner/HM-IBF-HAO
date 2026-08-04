use log::debug;
use mahf::{
    components::archive, problems::KnownOptimumProblem, state::common, Component, ExecResult,
    SingleObjectiveProblem, State,
};

use crate::problems::algorithm_design::{
    AlgorithmDesignProblem, EvaluationStatistics, GlobalEvaluationStatistics,
};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct NormalizeParents;

impl NormalizeParents {
    pub fn from_params() -> Self {
        Self
    }

    pub fn new<P: SingleObjectiveProblem + KnownOptimumProblem>(
    ) -> Box<dyn Component<AlgorithmDesignProblem<P>>> {
        Box::new(Self::from_params())
    }
}

impl<P: SingleObjectiveProblem + KnownOptimumProblem> Component<AlgorithmDesignProblem<P>>
    for NormalizeParents
{
    fn execute(
        &self,
        problem: &AlgorithmDesignProblem<P>,
        state: &mut State<AlgorithmDesignProblem<P>>,
    ) -> ExecResult<()> {
        let statistics = state.borrow::<GlobalEvaluationStatistics>();
        let mut populations = state.populations_mut();
        let children = populations.pop();
        let parents = populations.current_mut();

        for parent in parents {
            if let Ok(multi_evaluation) = parent.state().try_get_value::<EvaluationStatistics>() {
                let objective_value =
                    statistics.standardized_objective(&multi_evaluation, problem.penalty);
                parent.set_objective(objective_value);
            }
        }

        populations.push(children);

        Ok(())
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct NormalizeArchive;

impl NormalizeArchive {
    pub fn from_params() -> Self {
        Self
    }

    pub fn new<P: SingleObjectiveProblem + KnownOptimumProblem>(
    ) -> Box<dyn Component<AlgorithmDesignProblem<P>>> {
        Box::new(Self::from_params())
    }
}

impl<P: SingleObjectiveProblem + KnownOptimumProblem> Component<AlgorithmDesignProblem<P>>
    for NormalizeArchive
{
    fn execute(
        &self,
        problem: &AlgorithmDesignProblem<P>,
        state: &mut State<AlgorithmDesignProblem<P>>,
    ) -> ExecResult<()> {
        let statistics = state.borrow::<GlobalEvaluationStatistics>();
        let mut archive = state.borrow_mut::<archive::ElitistArchive<AlgorithmDesignProblem<P>>>();
        let elitists = archive.elitists_mut();

        if !elitists.is_empty() {
            for elitist in elitists {
                if let Ok(multi_evaluation) =
                    elitist.state().try_get_value::<EvaluationStatistics>()
                {
                    let objective_value =
                        statistics.standardized_objective(&multi_evaluation, problem.penalty);
                    elitist.set_objective(objective_value);
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct NormalizeBestIndividual;

impl NormalizeBestIndividual {
    pub fn from_params() -> Self {
        Self
    }

    pub fn new<P: SingleObjectiveProblem + KnownOptimumProblem>(
    ) -> Box<dyn Component<AlgorithmDesignProblem<P>>> {
        Box::new(Self::from_params())
    }
}

impl<P: SingleObjectiveProblem + KnownOptimumProblem> Component<AlgorithmDesignProblem<P>>
    for NormalizeBestIndividual
{
    fn execute(
        &self,
        problem: &AlgorithmDesignProblem<P>,
        state: &mut State<AlgorithmDesignProblem<P>>,
    ) -> ExecResult<()> {
        let mut best =
            state.borrow_value_mut::<common::BestIndividual<AlgorithmDesignProblem<P>>>();

        if let Some(best) = &mut *best {
            let statistics = state.borrow::<GlobalEvaluationStatistics>();

            if let Ok(multi_evaluation) = best.state().try_get_value::<EvaluationStatistics>() {
                let objective_value =
                    statistics.standardized_objective(&multi_evaluation, problem.penalty);
                debug!(
                    "Updated f(best) from {} to {}.",
                    best.objective().value(),
                    objective_value.value()
                );
                best.set_objective(objective_value);
            }
        }

        Ok(())
    }
}
