//! Resizing of solutions that migrate between islands of different dimensionality.
//!
//! When two islands operate at different dimensionalities, a migrant must be resized before
//! it can be inserted into the target population. Implementors of [`SolutionTransformer`]
//! provide that resizing logic.

use mahf::{Problem, Random};

/// Describes one migration between two islands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformRequest<'a> {
    /// The length of the migrating solution.
    pub source_dim: u32,
    /// The length the solution must have after the transform.
    pub target_dim: u32,
    /// The IRACE-tuned name of the transformation method.
    pub method: &'a str,
}

impl<'a> TransformRequest<'a> {
    /// Creates a new `TransformRequest`.
    ///
    /// # Arguments
    ///
    /// * `source_dim` - The length of the migrating solution.
    /// * `target_dim` - The required length after the transform.
    /// * `method` - The name of the transformation method.
    ///
    /// # Returns
    ///
    /// The request.
    pub fn new(source_dim: u32, target_dim: u32, method: &'a str) -> Self {
        Self {
            source_dim,
            target_dim,
            method,
        }
    }

    /// Returns whether source and target dimension agree, making the transform a no-op.
    pub fn is_identity(&self) -> bool {
        self.source_dim == self.target_dim
    }
}

/// Transforms solutions during migration between islands with different dimensions.
pub trait SolutionTransformer<P: Problem>: Send + Sync + 'static {
    /// Transforms `solution` from the request's source dimension to its target dimension.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem instance, for transforms that use problem geometry.
    /// * `solution` - The original solution encoding.
    /// * `request` - The source and target dimension plus the method name.
    /// * `rng` - Random number generator for stochastic transformations.
    ///
    /// # Returns
    ///
    /// The transformed solution, whose length is `request.target_dim`.
    fn transform(
        &self,
        problem: &P,
        solution: &P::Encoding,
        request: TransformRequest<'_>,
        rng: &mut Random,
    ) -> P::Encoding;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_with_equal_dimensions_is_an_identity() {
        assert!(TransformRequest::new(18, 18, "PCHIP").is_identity());
        assert!(!TransformRequest::new(18, 24, "PCHIP").is_identity());
    }
}
