//! Directed graph types used to encode island-model algorithms.

use std::{collections::HashMap, fmt::Debug};

pub use di::DiGraph;
use petgraph::graph::NodeIndex;

pub mod algorithms;
pub mod di;
pub mod set;

/// A path composed of nodes.
pub type NodePath<Ix> = Vec<NodeIndex<Ix>>;

/// A directed graph with integer node and edge weights.
pub type DiIntGraph = DiGraph<u32, u32>;

/// An edge that was removed from a graph, retaining its source node and weight.
#[derive(Debug, Clone, PartialEq)]
pub struct CutEdge<E, Ix> {
    pub(crate) source: NodeIndex<Ix>,
    pub(crate) weight: E,
}

impl<E, Ix> CutEdge<E, Ix> {
    /// Returns the source node of the edge.
    pub fn source(&self) -> NodeIndex<Ix>
    where
        Ix: Clone,
    {
        self.source.clone()
    }

    /// Rewrites the source node using `f`.
    ///
    /// # Arguments
    ///
    /// * `f` - Maps the current source node to the new one.
    pub fn map_source<F>(&mut self, f: F)
    where
        Ix: Clone,
        F: FnOnce(NodeIndex<Ix>) -> NodeIndex<Ix>,
    {
        let source = f(self.source.clone());
        self.source = source;
    }

    /// Returns a reference to the edge weight.
    pub fn weight(&self) -> &E {
        &self.weight
    }

    /// Returns a mutable reference to the edge weight.
    pub fn weight_mut(&mut self) -> &mut E {
        &mut self.weight
    }

    /// Converts the edge into its weight.
    pub fn into_weight(self) -> E {
        self.weight
    }
}

/// A [`DiGraph`] together with a mapping from origin-graph indices to its own indices.
#[derive(Debug, Clone)]
pub struct TranslatedDiGraph<N, E> {
    pub(crate) graph: DiGraph<N, E>,
    pub(crate) map: HashMap<NodeIndex<usize>, NodeIndex<usize>>,
}

impl<N, E> TranslatedDiGraph<N, E> {
    /// Returns a reference to the subgraph.
    pub fn graph(&self) -> &DiGraph<N, E> {
        &self.graph
    }

    /// Returns a mutable reference to the subgraph.
    pub fn graph_mut(&mut self) -> &mut DiGraph<N, E> {
        &mut self.graph
    }

    /// Converts the subgraph into a full graph, discarding the mapping.
    pub fn into_graph(self) -> DiGraph<N, E> {
        self.graph
    }

    /// Translates an index valid in the original graph into this subgraph's index space.
    ///
    /// # Arguments
    ///
    /// * `index` - The node index in the original graph.
    ///
    /// # Returns
    ///
    /// The corresponding index, or `None` if the node is not part of this subgraph.
    pub fn translate(&self, index: NodeIndex<usize>) -> Option<NodeIndex<usize>> {
        self.map.get(&index).copied()
    }
}

/// A fragment of a graph together with the edges that were cut off during fragmentation.
#[derive(Debug, Clone)]
pub struct DiGraphFragment<N, E> {
    /// The fragment itself.
    pub fragment: DiGraph<N, E>,
    /// The edges that originate in the fragment but pointed outside of it.
    pub cut_edges: Vec<CutEdge<E, usize>>,
}
