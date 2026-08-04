//! Trait alias for the class of problems solved by this crate's island builders.

use mahf::problems::LimitedVectorProblem;
use mahf::SingleObjectiveProblem;
use std::ops::Range;
use trait_set::trait_set;

trait_set! {
    pub trait RealValuedProblem = SingleObjectiveProblem + LimitedVectorProblem<Element=f64>;
}

/// Extension for problems whose search-space bounds depend on the working dimension.
///
/// `LimitedVectorProblem::domain()` always returns bounds for the *maximum* dimension.
/// Islands that run at a smaller IRACE-tuned dimension `D` must call
/// `domain_for_dimension(D)` so initialization samples correctly bounded positions.
///
/// # Stable-Rust coherence rules
///
/// There is NO blanket `impl<P: RealValuedProblem> DimensionAwareDomain for P`.
/// A blanket impl combined with a concrete `impl DimensionAwareDomain for HorizontalAlignment`
/// would be an overlapping impl pair that stable Rust rejects (E0119).
///
/// Instead:
/// * The trait supplies a **default method body** (`domain_for_dimension`) that does
///   safe slice/cycle of `self.domain()`.  Problems with position-uniform bounds get
///   this behaviour for free just by writing `impl DimensionAwareDomain for MyProblem {}`.
/// * Problems with position-dependent bounds (e.g. `HorizontalAlignment`) override the
///   method body to resample their geometry at the correct density.
/// * `RandomSpreadWithDimension` and helpers require `P: RealValuedProblem + DimensionAwareDomain`
///   so the correct bounds are always available without any dynamic dispatch.
pub trait DimensionAwareDomain: RealValuedProblem {
    /// Returns search-space bounds for exactly `dim` decision variables at their correct
    /// positions for this dimension.
    ///
    /// The default implementation slices (or cycles) `self.domain()`.  This is only
    /// correct when bounds don't vary with sampling position; override for geometric problems.
    fn domain_for_dimension(&self, dim: usize) -> Vec<Range<f64>> {
        let max_domain = self.domain();
        let max_len = max_domain.len();
        (0..dim).map(|i| max_domain[i % max_len].clone()).collect()
    }
}
