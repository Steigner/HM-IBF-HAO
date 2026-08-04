//! Cross-instance statistics used to standardize algorithm performance.
//!
//! Instances differ by orders of magnitude in objective scale, so raw performance values
//! cannot be summed. Every value is instead standardized against the median and median
//! deviation observed so far on the same instance.

use std::ops::Div;

use better_any::{Tid, TidAble};
use itertools::izip;
use mahf::{CustomState, SingleObjective};

use crate::utils::safe_div;

/// Per-instance performance of one configuration; `None` marks an instance where all runs failed.
pub type MultiProblemEvaluation = Vec<Option<f64>>;

/// A robust location/spread summary of the values observed for one instance.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Statistics {
    /// The median of the observed values.
    pub median: f64,
    /// The median absolute deviation of the observed values.
    pub deviation: f64,
    /// The number of observed values.
    pub len: usize,
}

/// Accumulates observations for a single instance and caches their [`Statistics`].
#[derive(Debug, Default)]
pub struct StatisticsTracker {
    /// All observations recorded so far.
    pub data: Vec<f64>,
    /// The cached summary, or `None` if [`StatisticsTracker::update`] has not run since the
    /// last insertion.
    pub statistics: Option<Statistics>,
}

impl StatisticsTracker {
    /// Creates an empty tracker.
    ///
    /// # Returns
    ///
    /// The tracker without observations.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a single observation, invalidating the cached statistics.
    ///
    /// # Arguments
    ///
    /// * `x` - The observation.
    pub fn push(&mut self, x: f64) {
        self.data.push(x);
        self.statistics = None;
    }

    /// Recomputes median and median deviation from all accumulated observations.
    ///
    /// Does nothing while no observation has been recorded.
    pub fn update(&mut self) {
        if self.data.is_empty() {
            return;
        }

        let len = self.data.len();
        self.data.sort_unstable_by(f64::total_cmp);
        let median = self.data[len / 2];

        let mut median_centered: Vec<_> = self.data.iter().map(|&x| (x - median).abs()).collect();
        median_centered.sort_unstable_by(|x, y| x.total_cmp(y));
        let deviation = median_centered[len / 2];

        self.statistics = Some(Statistics {
            median,
            deviation,
            len,
        });
    }

    /// Returns the cached statistics.
    ///
    /// # Returns
    ///
    /// The last computed summary, or `None` if [`StatisticsTracker::update`] has not run
    /// since the last insertion.
    pub fn get(&self) -> Option<Statistics> {
        self.statistics
    }
}

/// One [`StatisticsTracker`] per problem instance, stored in the outer optimization state.
#[derive(Debug, Tid)]
pub struct GlobalEvaluationStatistics(pub Vec<StatisticsTracker>);

impl GlobalEvaluationStatistics {
    /// Creates empty trackers for `num_problems` instances.
    ///
    /// # Arguments
    ///
    /// * `num_problems` - The number of problem instances.
    ///
    /// # Returns
    ///
    /// The tracker collection.
    pub fn new(num_problems: usize) -> Self {
        Self(
            (0..num_problems)
                .map(|_| StatisticsTracker::new())
                .collect(),
        )
    }

    /// Records one multi-instance evaluation, skipping instances where all runs failed.
    ///
    /// # Arguments
    ///
    /// * `multi_evaluation` - The per-instance performance of one configuration.
    pub fn push(&mut self, multi_evaluation: &MultiProblemEvaluation) {
        for (problem_statistics, evaluation) in self.0.iter_mut().zip(multi_evaluation) {
            if let Some(evaluation) = evaluation {
                problem_statistics.push(*evaluation);
            }
        }
    }

    /// Refreshes the cached statistics of every tracker.
    pub fn update(&mut self) {
        for problem_statistics in &mut self.0 {
            problem_statistics.update();
        }
    }

    /// Returns the cached statistics of every tracker, in instance order.
    pub fn get(&self) -> Vec<Option<Statistics>> {
        self.0.iter().map(StatisticsTracker::get).collect()
    }

    /// Standardizes a multi-instance evaluation against the *current* statistics.
    ///
    /// Use this to re-score individuals from earlier generations so they stay comparable
    /// with the current one. To score a freshly evaluated generation, use
    /// [`standardized_objective_with_stats`] with a pre-update snapshot instead.
    ///
    /// # Arguments
    ///
    /// * `multi_evaluation` - The per-instance performance to standardize.
    /// * `penalty` - The score substituted for missing statistics or failed instances.
    ///
    /// # Returns
    ///
    /// The mean standardized score across all instances.
    pub fn standardized_objective(
        &self,
        multi_evaluation: &MultiProblemEvaluation,
        penalty: f64,
    ) -> SingleObjective {
        standardized_objective_with_stats(&self.get(), multi_evaluation, penalty)
    }
}

impl CustomState<'_> for GlobalEvaluationStatistics {}

/// Standardizes a multi-instance evaluation against an external statistics snapshot.
///
/// Pass statistics computed *before* the current generation was added, so an individual is
/// never part of its own normalization baseline.
///
/// # Arguments
///
/// * `stats` - The per-instance statistics snapshot; `None` entries fall back to `penalty`.
/// * `multi_evaluation` - The per-instance performance to standardize.
/// * `penalty` - The score substituted for missing statistics or failed instances.
///
/// # Returns
///
/// The mean standardized score across all instances, or the default objective if the mean
/// is not a valid objective value (for example `NaN`).
pub fn standardized_objective_with_stats(
    stats: &[Option<Statistics>],
    multi_evaluation: &MultiProblemEvaluation,
    penalty: f64,
) -> SingleObjective {
    izip!(stats, multi_evaluation)
        .map(|(problem_statistics, problem_evaluation)| {
            match (problem_evaluation, problem_statistics) {
                (Some(evaluation), Some(statistics)) => {
                    safe_div(evaluation - statistics.median, statistics.deviation)
                }
                // No prior statistics (first generation) or failed run - treat as penalty.
                _ => penalty,
            }
        })
        .sum::<f64>()
        .div(multi_evaluation.len() as f64)
        .try_into()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_caches_statistics_only_after_update() {
        let mut tracker = StatisticsTracker::new();
        for value in [1.0, 2.0, 3.0] {
            tracker.push(value);
        }
        assert_eq!(tracker.get(), None, "cache is invalid before update");

        tracker.update();

        let statistics = tracker.get().unwrap();
        assert_eq!(statistics.median, 2.0);
        assert_eq!(statistics.deviation, 1.0);
        assert_eq!(statistics.len, 3);
    }

    #[test]
    fn pushing_invalidates_the_cached_statistics() {
        let mut tracker = StatisticsTracker::new();
        for value in [1.0, 2.0, 3.0] {
            tracker.push(value);
        }
        tracker.update();

        tracker.push(4.0);

        assert_eq!(tracker.get(), None);
    }

    #[test]
    fn update_on_an_empty_tracker_is_a_noop() {
        let mut tracker = StatisticsTracker::new();
        tracker.update();
        assert_eq!(tracker.get(), None);
    }

    #[test]
    fn global_statistics_ignore_failed_instances() {
        let mut statistics = GlobalEvaluationStatistics::new(2);

        statistics.push(&vec![Some(1.0), None]);
        statistics.push(&vec![Some(3.0), Some(7.0)]);
        statistics.update();

        let summaries = statistics.get();
        assert_eq!(summaries[0].unwrap().len, 2);
        assert_eq!(summaries[1].unwrap().len, 1);
    }

    #[test]
    fn standardization_maps_the_median_to_zero() {
        let stats = vec![Some(Statistics {
            median: 10.0,
            deviation: 2.0,
            len: 5,
        })];

        let objective = standardized_objective_with_stats(&stats, &vec![Some(10.0)], 100.0);

        assert_eq!(objective.value(), 0.0);
    }

    #[test]
    fn standardization_scales_by_the_median_deviation() {
        let stats = vec![Some(Statistics {
            median: 10.0,
            deviation: 2.0,
            len: 5,
        })];

        let objective = standardized_objective_with_stats(&stats, &vec![Some(14.0)], 100.0);

        assert_eq!(objective.value(), 2.0);
    }

    #[test]
    fn standardization_falls_back_to_the_penalty() {
        let stats = vec![
            None,
            Some(Statistics {
                median: 0.0,
                deviation: 1.0,
                len: 1,
            }),
        ];

        // Instance 0 has no statistics, instance 1 failed - both contribute the penalty.
        let objective = standardized_objective_with_stats(&stats, &vec![Some(5.0), None], 100.0);

        assert_eq!(objective.value(), 100.0);
    }

    #[test]
    fn standardization_survives_a_zero_deviation() {
        let stats = vec![Some(Statistics {
            median: 10.0,
            deviation: 0.0,
            len: 5,
        })];

        let objective = standardized_objective_with_stats(&stats, &vec![Some(14.0)], 100.0);

        assert!(
            objective.value().is_finite(),
            "safe_div must not produce an infinite score"
        );
        assert_eq!(objective.value(), 4.0);
    }
}
