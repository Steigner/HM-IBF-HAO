//! Data types representing the migrations (edges) of an algorithm graph.
//!
//! Islands themselves are plain `Component`s built by
//! [`crate::problems::algorithm_design::builder::MetaheuristicIslandBuilder`].

use derivative::Derivative;
use mahf::{Component, Condition, Problem};
use serde::Serialize;

/// A transfer of individuals between islands.
#[derive(Serialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = ""))]
pub struct Migration<P: Problem> {
    /// Decides when to transfer individuals.
    pub condition: Box<dyn Condition<P>>,
    /// Selects the individuals that should be transferred.
    pub selection: Box<dyn Component<P>>,
    /// Inserts the individuals into the target population.
    pub replacement: Box<dyn Component<P>>,

    /// Transformation method for dimension changes during migration (IRACE-tuned).
    pub transform_method: Option<String>,

    /// Source island dimension (IRACE-tuned).
    pub source_dimension: Option<u32>,

    /// Target island dimension (IRACE-tuned).
    pub target_dimension: Option<u32>,
}
