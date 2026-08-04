//! Directed graph with arbitrary node and edge weights.
//!
//! The type is split across submodules to keep each file focused:
//!
//! * this module - the type itself, accessors and formatting,
//! * [`generate`] - randomized construction of graph topologies,
//! * [`topology`] - paths, connected components, fragmentation and merging.

use std::{
    fmt::{Debug, Display, Formatter},
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use petgraph::{
    algo::is_isomorphic_matching,
    data::{Element, FromElements},
    dot::Dot,
    graph::{
        EdgeIndex, EdgeIndices, EdgeReferences, EdgeWeights, EdgeWeightsMut, NodeIndex,
        NodeIndices, NodeReferences, NodeWeights, NodeWeightsMut,
    },
    Directed, Graph,
};
use serde::{Deserialize, Serialize};

mod generate;
mod topology;

/// A directed graph with arbitrary node and edge weights.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct DiGraph<N, E> {
    graph: Graph<N, E, Directed, usize>,
}

impl<N, E> DiGraph<N, E> {
    /// Constructs a new, empty `DiGraph`.
    ///
    /// # Returns
    ///
    /// The empty graph.
    pub fn new() -> Self {
        Self {
            graph: Graph::default(),
        }
    }

    /// Constructs a new `DiGraph` from [`Element`]s.
    ///
    /// # Arguments
    ///
    /// * `iterable` - The nodes and edges, in `petgraph`'s element order.
    ///
    /// # Returns
    ///
    /// The constructed graph.
    pub fn from_elements<I>(iterable: I) -> Self
    where
        I: IntoIterator<Item = Element<N, E>>,
    {
        Self {
            graph: Graph::from_elements(iterable),
        }
    }

    /// Returns the number of nodes (vertices) in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Returns whether an edge from `a` to `b` exists.
    ///
    /// # Arguments
    ///
    /// * `a` - The source node.
    /// * `b` - The target node.
    ///
    /// # Returns
    ///
    /// `true` if the directed edge exists.
    pub fn contains_edge(&self, a: NodeIndex<usize>, b: NodeIndex<usize>) -> bool {
        self.graph.contains_edge(a, b)
    }

    /// Returns whether the node at `index` has any neighbour, ignoring edge direction.
    ///
    /// # Arguments
    ///
    /// * `index` - The node to inspect.
    ///
    /// # Returns
    ///
    /// `true` if at least one incident edge exists.
    pub fn has_neighbors(&self, index: NodeIndex<usize>) -> bool {
        self.graph.neighbors_undirected(index).next().is_some()
    }

    /// Adds a node with associated data `weight` to the graph.
    ///
    /// # Arguments
    ///
    /// * `weight` - The node weight.
    ///
    /// # Returns
    ///
    /// The index of the new node.
    pub fn add_node(&mut self, weight: N) -> NodeIndex<usize> {
        self.graph.add_node(weight)
    }

    /// Removes a node from the graph.
    ///
    /// # Arguments
    ///
    /// * `index` - The node to remove.
    ///
    /// # Returns
    ///
    /// The removed node's weight, or `None` if it does not exist.
    pub fn remove_node(&mut self, index: NodeIndex<usize>) -> Option<N> {
        self.graph.remove_node(index)
    }

    /// Adds an edge from `a` to `b`, updating the weight if the edge already exists.
    ///
    /// # Arguments
    ///
    /// * `a` - The source node.
    /// * `b` - The target node.
    /// * `weight` - The edge weight.
    ///
    /// # Returns
    ///
    /// The index of the new or updated edge.
    pub fn add_edge(
        &mut self,
        a: NodeIndex<usize>,
        b: NodeIndex<usize>,
        weight: E,
    ) -> EdgeIndex<usize> {
        self.graph.update_edge(a, b, weight)
    }

    /// Removes an edge from the graph.
    ///
    /// # Arguments
    ///
    /// * `index` - The edge to remove.
    ///
    /// # Returns
    ///
    /// The removed edge's weight, or `None` if it does not exist.
    pub fn remove_edge(&mut self, index: EdgeIndex<usize>) -> Option<E> {
        self.graph.remove_edge(index)
    }

    /// Returns a reference to a node weight.
    ///
    /// # Arguments
    ///
    /// * `index` - The node to inspect.
    ///
    /// # Returns
    ///
    /// The weight, or `None` if the node does not exist.
    pub fn node_weight(&self, index: NodeIndex<usize>) -> Option<&N> {
        self.graph.node_weight(index)
    }

    /// Returns a mutable reference to a node weight.
    ///
    /// # Arguments
    ///
    /// * `index` - The node to inspect.
    ///
    /// # Returns
    ///
    /// The weight, or `None` if the node does not exist.
    pub fn node_weight_mut(&mut self, index: NodeIndex<usize>) -> Option<&mut N> {
        self.graph.node_weight_mut(index)
    }

    /// Returns a reference to an edge weight.
    ///
    /// # Arguments
    ///
    /// * `index` - The edge to inspect.
    ///
    /// # Returns
    ///
    /// The weight, or `None` if the edge does not exist.
    pub fn edge_weight(&self, index: EdgeIndex<usize>) -> Option<&E> {
        self.graph.edge_weight(index)
    }

    /// Returns a mutable reference to an edge weight.
    ///
    /// # Arguments
    ///
    /// * `index` - The edge to inspect.
    ///
    /// # Returns
    ///
    /// The weight, or `None` if the edge does not exist.
    pub fn edge_weight_mut(&mut self, index: EdgeIndex<usize>) -> Option<&mut E> {
        self.graph.edge_weight_mut(index)
    }

    /// Returns an iterator over the node indices of the graph.
    pub fn node_indices(&self) -> NodeIndices<usize> {
        self.graph.node_indices()
    }

    /// Returns an iterator over the edge indices of the graph.
    pub fn edge_indices(&self) -> EdgeIndices<usize> {
        self.graph.edge_indices()
    }

    /// Returns an iterator over all node weights, in node index order.
    pub fn node_weights(&self) -> NodeWeights<'_, N, usize> {
        self.graph.node_weights()
    }

    /// Returns a mutable iterator over all node weights, in node index order.
    pub fn node_weights_mut(&mut self) -> NodeWeightsMut<'_, N, usize> {
        self.graph.node_weights_mut()
    }

    /// Returns an iterator over all nodes with their indices, in node index order.
    pub fn node_references(&self) -> NodeReferences<'_, N, usize> {
        self.graph.node_references()
    }

    /// Converts the graph into its node and edge weights.
    ///
    /// # Returns
    ///
    /// The node weights and the edge weights, each in their original index order.
    pub fn into_node_edge_weights(self) -> (Vec<N>, Vec<E>) {
        let (nodes, edges) = self.graph.into_nodes_edges();

        let nodes = nodes.into_iter().map(|node| node.weight).collect();
        let edges = edges.into_iter().map(|edge| edge.weight).collect();

        (nodes, edges)
    }

    /// Returns an iterator over all edge weights, in edge index order.
    pub fn edge_weights(&self) -> EdgeWeights<'_, E, usize> {
        self.graph.edge_weights()
    }

    /// Returns a mutable iterator over all edge weights, in edge index order.
    pub fn edge_weights_mut(&mut self) -> EdgeWeightsMut<'_, E, usize> {
        self.graph.edge_weights_mut()
    }

    /// Returns an iterator over all edges, in edge index order.
    pub fn edge_references(&self) -> EdgeReferences<'_, E, usize> {
        self.graph.edge_references()
    }

    /// Returns the source and target nodes of an edge.
    ///
    /// # Arguments
    ///
    /// * `index` - The edge to inspect.
    ///
    /// # Returns
    ///
    /// The `(source, target)` pair, or `None` if the edge does not exist.
    pub fn edge_endpoints(
        &self,
        index: EdgeIndex<usize>,
    ) -> Option<(NodeIndex<usize>, NodeIndex<usize>)> {
        self.graph.edge_endpoints(index)
    }

    /// Creates a new `DiGraph` by mapping node and edge weights to new values.
    ///
    /// The resulting graph has the same structure and the same indices as `self`.
    ///
    /// # Arguments
    ///
    /// * `node_map` - Maps each node index and weight to a new weight.
    /// * `edge_map` - Maps each edge index and weight to a new weight.
    ///
    /// # Returns
    ///
    /// The mapped graph.
    pub fn map<F, G, N2, E2>(&self, node_map: F, edge_map: G) -> DiGraph<N2, E2>
    where
        F: FnMut(NodeIndex<usize>, &N) -> N2,
        G: FnMut(EdgeIndex<usize>, &E) -> E2,
    {
        DiGraph {
            graph: self.graph.map(node_map, edge_map),
        }
    }

    /// Like [`DiGraph::map`], but the mapping functions may fail.
    ///
    /// # Arguments
    ///
    /// * `node_map` - Maps each node index and weight to a new weight, or fails.
    /// * `edge_map` - Maps each edge index and weight to a new weight, or fails.
    ///
    /// # Returns
    ///
    /// The mapped graph.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by either mapping function.
    pub fn try_map<F, G, N2, E2, Error>(
        &self,
        node_map: F,
        edge_map: G,
    ) -> Result<DiGraph<N2, E2>, Error>
    where
        F: FnMut(NodeIndex<usize>, &N) -> Result<N2, Error>,
        G: FnMut(EdgeIndex<usize>, &E) -> Result<E2, Error>,
    {
        self.graph
            .try_map(node_map, edge_map)
            .map(|graph| DiGraph { graph })
    }
}

impl<N, E> DiGraph<N, E>
where
    N: Display,
    E: Display,
{
    /// Writes the graph to `path` in `graphviz` `dot` format.
    ///
    /// # Arguments
    ///
    /// * `path` - The destination file, which is created or truncated.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written.
    pub fn to_dot(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let dot = Dot::new(&self.graph);
        let file = File::create(path.as_ref())?;
        let mut buffer = BufWriter::new(file);
        writeln!(&mut buffer, "{dot}")?;
        Ok(())
    }

    /// Renders the graph in `graphviz` `dot` format.
    ///
    /// # Returns
    ///
    /// The `dot` representation.
    pub fn to_dot_string(&self) -> String {
        Dot::new(&self.graph).to_string()
    }
}

impl<N, E> Display for DiGraph<N, E>
where
    N: Display,
    E: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let dot = Dot::new(&self.graph);
        dot.fmt(f)
    }
}

impl<N, E> PartialEq for DiGraph<N, E>
where
    N: PartialEq,
    E: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        is_isomorphic_matching(&self.graph, &other.graph, |a, b| a == b, |a, b| a == b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a directed path `0 -> 1 -> 2` with node weights equal to their index.
    fn path_graph() -> DiGraph<u32, u32> {
        let mut graph = DiGraph::new();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        graph.add_edge(a, b, 10);
        graph.add_edge(b, c, 11);
        graph
    }

    #[test]
    fn counts_reflect_inserted_nodes_and_edges() {
        let graph = path_graph();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn add_edge_updates_the_weight_of_an_existing_edge() {
        let mut graph = path_graph();
        let a = NodeIndex::new(0);
        let b = NodeIndex::new(1);

        let index = graph.add_edge(a, b, 99);

        assert_eq!(graph.edge_count(), 2, "no parallel edge must be created");
        assert_eq!(graph.edge_weight(index), Some(&99));
    }

    #[test]
    fn contains_edge_respects_direction() {
        let graph = path_graph();
        assert!(graph.contains_edge(NodeIndex::new(0), NodeIndex::new(1)));
        assert!(!graph.contains_edge(NodeIndex::new(1), NodeIndex::new(0)));
    }

    #[test]
    fn has_neighbors_ignores_direction() {
        let mut graph = path_graph();
        let isolated = graph.add_node(3);

        assert!(graph.has_neighbors(NodeIndex::new(2)), "target of an edge");
        assert!(!graph.has_neighbors(isolated));
    }

    #[test]
    fn map_preserves_structure_and_indices() {
        let graph = path_graph();

        let mapped = graph.map(|_, &w| w.to_string(), |_, &w| w * 2);

        assert_eq!(mapped.node_count(), graph.node_count());
        assert_eq!(mapped.edge_count(), graph.edge_count());
        assert_eq!(
            mapped.node_weight(NodeIndex::new(1)),
            Some(&"1".to_string())
        );
        assert_eq!(mapped.edge_weight(EdgeIndex::new(0)), Some(&20));
    }

    #[test]
    fn try_map_propagates_the_first_error() {
        let graph = path_graph();

        let result: Result<DiGraph<u32, u32>, &str> = graph.try_map(
            |_, &w| if w == 1 { Err("boom") } else { Ok(w) },
            |_, &w| Ok(w),
        );

        assert_eq!(result.err(), Some("boom"));
    }

    #[test]
    fn into_node_edge_weights_keeps_index_order() {
        let (nodes, edges) = path_graph().into_node_edge_weights();
        assert_eq!(nodes, vec![0, 1, 2]);
        assert_eq!(edges, vec![10, 11]);
    }

    #[test]
    fn equality_is_isomorphism_on_weights() {
        let mut relabelled = DiGraph::new();
        let c = relabelled.add_node(2);
        let b = relabelled.add_node(1);
        let a = relabelled.add_node(0);
        relabelled.add_edge(a, b, 10);
        relabelled.add_edge(b, c, 11);

        assert_eq!(path_graph(), relabelled);
    }

    #[test]
    fn dot_rendering_contains_every_node_weight() {
        let dot = path_graph().to_dot_string();
        for weight in ["0", "1", "2"] {
            assert!(dot.contains(weight), "missing {weight} in {dot}");
        }
    }
}
