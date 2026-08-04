use mahf::Problem;

use crate::graph::DiIntGraph;

pub mod algorithm_design;

pub mod evaluate;

/// A directed graph-based optimization problem.
///
/// This trait extends the [`Problem`] trait and represents an optimization problem
/// whose solutions are encoded as directed graphs with nodes and edges holding an `u32`.
///
/// [`Element`]: VectorProblem::Element
pub trait DirectedGraphProblem: Problem<Encoding = DiIntGraph> {
    /// The number of node types.
    fn node_types(&self) -> Vec<u32>;

    /// The number of edge types.
    fn edge_types(&self) -> Vec<u32>;
}

/// A problem with constraints.
pub trait ConstrainedProblem: Problem {
    /// Checks if the solution is feasible.
    fn feasible(&self, solution: &Self::Encoding) -> bool;
}
