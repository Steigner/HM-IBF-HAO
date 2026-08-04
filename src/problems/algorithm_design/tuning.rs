//! IRACE-based parameter tuning for algorithm configurations.

use std::{iter::repeat, sync::Arc};

use eyre::ContextCompat;
use irace_rs::{
    multi_irace, param_space::ParamSpace, scenario::Scenario, DistributedInstance, Experiment, Run,
    TargetRunner,
};
use itertools::izip;
use log::{log_enabled, trace};
use mahf::{
    identifier::Global,
    params::Params,
    problems::{Evaluate, KnownOptimumProblem},
    state::common::Evaluator,
    ExecResult, Random, SingleObjective, SingleObjectiveProblem,
};

use crate::{
    components::{island::MigrationTransformer, transform::SolutionTransformer},
    problems::algorithm_design::{builder::MetaheuristicBuilder, performance::PerformanceMeasure},
};

/// Builds an IRACE target runner that evaluates one configuration on one instance.
///
/// The runner builds a `Configuration` from IRACE-provided `Params`, runs it `num_repetitions`
/// times, and returns the aggregate performance as a single objective value.
pub fn make_target_runner<P: SingleObjectiveProblem + Send + Sync>(
    builder: impl MetaheuristicBuilder<P>,
    performance_measure: Box<dyn PerformanceMeasure<P>>,
    num_repetitions: u32,
    transformer: Option<Arc<dyn SolutionTransformer<P>>>,
) -> impl TargetRunner<DistributedInstance<P>> + Clone {
    move |_: &Scenario,
          experiment: Experiment<DistributedInstance<P>>|
          -> ExecResult<SingleObjective> {
        let config = builder(experiment.params)?;
        let instance = experiment.instance.wrap_err("missing instance")?;
        let problem = instance.problem();
        let transformer_clone = transformer.clone();

        let mut rng = Random::new(experiment.seed);
        let states: Vec<_> = rng
            .iter_children()
            .take(num_repetitions as usize)
            .map(|rng| {
                let evaluator = instance.evaluator();
                let t = transformer_clone.clone();
                config.optimize_with(problem, |state| {
                    state.insert(Evaluator::<_, Global>::from(evaluator));
                    state.insert(rng);
                    // Insert transformer for dimension-aware migration during tuning
                    if let Some(ref transformer) = t {
                        state.insert(MigrationTransformer(transformer.clone()));
                    }
                    Ok(())
                })
            })
            .collect::<Result<_, _>>()?;

        let performance = performance_measure.measure_finite(&states);

        if log_enabled!(log::Level::Trace) {
            trace!(
                "Finished experiment {} on {} with an {} of {}",
                experiment.id,
                problem.name(),
                performance_measure.name(),
                performance
            );
        }

        Ok(SingleObjective::try_from(performance).unwrap_or_default())
    }
}

/// Tunes multiple algorithm builders in parallel using IRACE, one per graph individual.
///
/// Returns the best `Params` found for each builder in the same order as `builders`.
#[allow(clippy::too_many_arguments)]
pub fn par_tune<P, O>(
    builders: Vec<impl MetaheuristicBuilder<P>>,
    param_spaces: Vec<ParamSpace>,
    scenario: Arc<Scenario>,
    problems: &[Arc<P>],
    evaluators: Vec<O>,
    performance_measure: Box<dyn PerformanceMeasure<P>>,
    num_repetitions: u32,
    num_jobs: usize,
    global_seed: u32,
    transformer: Option<Arc<dyn SolutionTransformer<P>>>,
) -> ExecResult<Vec<Params>>
where
    P: SingleObjectiveProblem + KnownOptimumProblem + Send + Sync,
    O: Evaluate<Problem = P> + Clone + 'static,
{
    let param_spaces: Vec<_> = param_spaces.into_iter().map(Arc::new).collect();

    // Construct the evaluation methods for irace.
    let target_runners: Vec<_> = builders
        .iter()
        .cloned()
        .map(|builder| {
            make_target_runner(
                builder,
                performance_measure.clone(),
                num_repetitions,
                transformer.clone(),
            )
        })
        .collect();

    // Prepare the problem instances for irace.
    let instances: Vec<_> = problems
        .iter()
        .zip(evaluators)
        .map(|(problem, evaluator)| DistributedInstance::new(problem.clone(), evaluator))
        .collect();

    // Prepare runs.
    let runs = izip!(
        target_runners,
        repeat(instances),
        repeat(scenario),
        param_spaces
    )
    .map(|(target_runner, instances, scenario, param_space)| {
        Run::new(target_runner, instances, scenario, param_space)
    });

    // Use `multi_irace` to find optimal parameters for each graph in parallel.
    Ok(multi_irace(runs, num_jobs, Some(global_seed))?
        .into_iter()
        .map(|all_params| all_params.into_iter().next().unwrap())
        .collect())
}
