//! Small numeric and sampling helpers shared across the framework.

use std::{collections::HashSet, hash::Hash};

use rand::{distributions::uniform::SampleUniform, seq::IteratorRandom, Rng};

/// Samples one element of `a` that is not contained in `b`.
///
/// # Arguments
///
/// * `a` - Candidate elements to sample from.
/// * `b` - Elements to exclude from the sample.
/// * `rng` - Random number generator used for the draw.
///
/// # Returns
///
/// A uniformly sampled element of `a \ b`, or `None` if that set is empty.
pub fn gen_distinct<T, I1, I2, R>(a: I1, b: I2, rng: &mut R) -> Option<T>
where
    T: SampleUniform + Hash + Eq + Clone,
    I1: IntoIterator<Item = T>,
    I2: IntoIterator<Item = T>,
    R: Rng + ?Sized,
{
    let all: HashSet<_> = a.into_iter().collect();
    let exclude: HashSet<_> = b.into_iter().collect();
    let difference = all.difference(&exclude);

    difference.choose(rng).cloned()
}

/// Divides `x` by `y`, returning `x` unchanged when `y` is zero.
///
/// Used for standardization where a zero spread must not produce a non-finite score.
///
/// # Arguments
///
/// * `x` - The dividend.
/// * `y` - The divisor.
///
/// # Returns
///
/// `x / y`, or `x` if `y == 0.0`.
pub fn safe_div(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        x
    } else {
        x / y
    }
}

/// Returns the median of `xs`, i.e. the upper of the two central values for even lengths.
///
/// # Arguments
///
/// * `xs` - The samples to reduce.
///
/// # Returns
///
/// The median, or `None` if `xs` is empty.
pub fn median(xs: &[f64]) -> Option<f64> {
    percentile(xs, 0.5)
}

/// Returns the `p`-quantile of `xs` using nearest-rank selection.
///
/// # Arguments
///
/// * `xs` - The samples to reduce.
/// * `p` - The quantile in `[0, 1]`; values outside the range are clamped.
///
/// # Returns
///
/// The selected sample, or `None` if `xs` is empty.
pub fn percentile(xs: &[f64], p: f64) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }

    let mut xs = xs.to_vec();
    xs.sort_unstable_by(|x, y| x.total_cmp(y));

    // Nearest-rank selection; the clamp keeps `p == 1.0` inside the slice.
    let rank = (xs.len() as f64 * p.clamp(0.0, 1.0)) as usize;
    Some(xs[rank.min(xs.len() - 1)])
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    #[test]
    fn gen_distinct_never_returns_excluded_elements() {
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..32 {
            let sample = gen_distinct(0..5, [1, 2, 3], &mut rng).unwrap();
            assert!(sample == 0 || sample == 4);
        }
    }

    #[test]
    fn gen_distinct_returns_none_when_everything_is_excluded() {
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(gen_distinct(0..3, 0..3, &mut rng), None);
    }

    #[test]
    fn safe_div_returns_dividend_for_zero_divisor() {
        assert_eq!(safe_div(4.0, 2.0), 2.0);
        assert_eq!(safe_div(4.0, 0.0), 4.0);
    }

    #[test]
    fn reductions_return_none_for_empty_slices() {
        assert_eq!(median(&[]), None);
        assert_eq!(percentile(&[], 0.5), None);
    }

    #[test]
    fn the_median_picks_the_upper_central_value_for_even_lengths() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(3.0));
        assert_eq!(median(&[1.0, 2.0, 3.0]), Some(2.0));
        assert_eq!(median(&[7.0]), Some(7.0));
    }

    #[test]
    fn percentile_stays_in_bounds_for_the_full_range() {
        let xs = [1.0, 2.0, 3.0];
        assert_eq!(percentile(&xs, 0.0), Some(1.0));
        assert_eq!(percentile(&xs, 1.0), Some(3.0));
        // Out-of-range quantiles are clamped rather than panicking.
        assert_eq!(percentile(&xs, 2.0), Some(3.0));
        assert_eq!(percentile(&xs, -1.0), Some(1.0));
    }

    #[test]
    fn percentile_orders_non_finite_samples_deterministically() {
        // `total_cmp` gives NaN a defined position instead of panicking.
        assert_eq!(percentile(&[f64::NAN, 1.0, 2.0], 0.0), Some(1.0));
    }
}
