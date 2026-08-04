//! Performance measures that condense repeated runs of one configuration into a single score.

use dyn_clone::DynClone;
use mahf::{SingleObjectiveProblem, State};

use crate::utils::median;

/// Condenses the states of repeated algorithm runs into a single comparable score.
///
/// Lower scores are always better.
pub trait PerformanceMeasure<P: SingleObjectiveProblem>: DynClone + Send + Sync + 'static {
    /// Returns the short name of the measure, used for logging.
    fn name(&self) -> &'static str;

    /// Measures the performance of the given runs.
    ///
    /// # Arguments
    ///
    /// * `states` - The final states of the repeated runs.
    ///
    /// # Returns
    ///
    /// The score, or `None` if no run produced a usable result.
    fn measure(&self, states: &[State<P>]) -> Option<f64>;

    /// Measures the performance of the given runs, substituting a penalty for failures.
    ///
    /// IRACE cannot represent "no result", so this variant must always return a finite
    /// score that ranks failed runs behind every successful one.
    ///
    /// # Arguments
    ///
    /// * `states` - The final states of the repeated runs.
    ///
    /// # Returns
    ///
    /// The score, or a penalty value if no run produced a usable result.
    fn measure_finite(&self, states: &[State<P>]) -> f64;
}

dyn_clone::clone_trait_object!(<P: SingleObjectiveProblem> PerformanceMeasure<P>);

/// Median of the best objective values reached across the repeated runs.
#[derive(Default, Clone)]
pub struct MedianBestObjectiveValue;

impl MedianBestObjectiveValue {
    /// Creates a new `MedianBestObjectiveValue`.
    ///
    /// # Returns
    ///
    /// The measure.
    pub fn new() -> Self {
        Self
    }
}

impl<P: SingleObjectiveProblem> PerformanceMeasure<P> for MedianBestObjectiveValue {
    fn name(&self) -> &'static str {
        "MBO"
    }

    fn measure(&self, states: &[State<P>]) -> Option<f64> {
        let objective_values: Vec<_> = states
            .iter()
            .map(|state| state.best_objective_value().map(|o| o.value()))
            .collect::<Option<_>>()?;
        median(&objective_values)
    }

    fn measure_finite(&self, states: &[State<P>]) -> f64 {
        // A run without a best individual must not abort tuning; rank it last instead.
        self.measure(states).unwrap_or(f64::MAX)
    }
}
