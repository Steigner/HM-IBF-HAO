//! Path, connectivity, fragmentation and merging operations on [`DiGraph`].

use std::collections::{HashMap, HashSet};

use petgraph::{
    algo::all_simple_paths,
    graph::NodeIndex,
    prelude::EdgeRef,
    visit::{depth_first_search, Control, DfsEvent},
    Graph,
};
use rand::Rng;

use crate::graph::{
    di::DiGraph,
    set::{Clustering, DisjointSets},
    CutEdge, DiGraphFragment, NodePath, TranslatedDiGraph,
};

impl<N, E> DiGraph<N, E> {
    /// Returns a path with the fewest edges between `from` and `to`.
    ///
    /// Node and edge weights are ignored.
    ///
    /// # Arguments
    ///
    /// * `from` - The start node.
    /// * `to` - The end node.
    ///
    /// # Returns
    ///
    /// The node sequence of a shortest path, or `None` if `to` is unreachable from `from`.
    pub fn shortest_path(
        &self,
        from: NodeIndex<usize>,
        to: NodeIndex<usize>,
    ) -> Option<NodePath<usize>> {
        all_simple_paths::<Vec<_>, _>(&self.graph, from, to, 0, None).min_by_key(Vec::len)
    }

    /// Removes edges until no path leads from `from` to `to`.
    ///
    /// On each round one random edge of a currently shortest path is removed. Paths from
    /// `to` back to `from` are not cut.
    ///
    /// # Arguments
    ///
    /// * `from` - The start node.
    /// * `to` - The end node.
    /// * `rng` - Random number generator used to pick which edge of a path to cut.
    ///
    /// # Returns
    ///
    /// The removed edges, each keeping its source node and weight.
    pub fn break_paths<R>(
        &mut self,
        from: NodeIndex<usize>,
        to: NodeIndex<usize>,
        rng: &mut R,
    ) -> Vec<CutEdge<E, usize>>
    where
        R: Rng + ?Sized,
    {
        let mut cut_edges = Vec::new();

        while let Some(path) = self.shortest_path(from, to) {
            // A path with a single node carries no edge to cut; stopping here prevents an
            // empty sampling range and an otherwise unbounded loop.
            let Some(last_edge) = path.len().checked_sub(1).filter(|&len| len > 0) else {
                break;
            };

            let index = rng.gen_range(0..last_edge);
            let (a, b) = (path[index], path[index + 1]);
            let Some(edge) = self.graph.find_edge(a, b) else {
                break;
            };
            let Some(weight) = self.graph.remove_edge(edge) else {
                break;
            };
            cut_edges.push(CutEdge { source: a, weight });
        }

        cut_edges
    }

    /// Reconnects a previously cut edge to a new target.
    ///
    /// # Arguments
    ///
    /// * `edge` - The cut edge, carrying its original source and weight.
    /// * `to` - The new target node.
    ///
    /// # Returns
    ///
    /// The index of the restored edge.
    pub fn restore_edge(
        &mut self,
        edge: CutEdge<E, usize>,
        to: NodeIndex<usize>,
    ) -> petgraph::graph::EdgeIndex<usize> {
        let source = edge.source();
        self.add_edge(source, to, edge.into_weight())
    }

    /// Groups the nodes into weakly connected components.
    ///
    /// # Returns
    ///
    /// A disjoint-set structure in which each set is one weakly connected component.
    pub fn node_sets(&self) -> DisjointSets<NodeIndex<usize>> {
        let mut sets = DisjointSets::new(self.graph.node_count());
        for edge in self.graph.edge_references() {
            sets.union(edge.source(), edge.target());
        }
        sets
    }

    /// Returns the number of weakly connected components.
    pub fn num_components(&self) -> usize {
        self.node_sets().count()
    }

    /// Returns the weakly connected components as a clustering.
    pub fn connected_components(&self) -> Clustering<NodeIndex<usize>> {
        self.node_sets().to_clustering()
    }

    /// Returns the nodes reachable from `source` following edge direction.
    ///
    /// # Arguments
    ///
    /// * `source` - The node to start the depth-first search from.
    ///
    /// # Returns
    ///
    /// The discovered nodes, including `source` itself.
    pub fn connected_component(&self, source: NodeIndex<usize>) -> HashSet<NodeIndex<usize>> {
        let mut component = HashSet::new();
        depth_first_search(&self.graph, Some(source), |event| {
            if let DfsEvent::Discover(node, _) = event {
                component.insert(node);
            }
            Control::<()>::Continue
        });
        component
    }

    /// Returns the nodes reachable from both `a` and `b`.
    ///
    /// # Arguments
    ///
    /// * `a` - The first source node.
    /// * `b` - The second source node.
    ///
    /// # Returns
    ///
    /// The intersection of both reachable sets.
    pub fn overlap(&self, a: NodeIndex<usize>, b: NodeIndex<usize>) -> HashSet<NodeIndex<usize>> {
        let ca = self.connected_component(a);
        let cb = self.connected_component(b);
        ca.intersection(&cb).cloned().collect()
    }

    /// Splits the graph into its weakly connected components.
    ///
    /// # Returns
    ///
    /// One subgraph per component, each able to translate indices of the original graph.
    pub fn split_components(&self) -> Vec<TranslatedDiGraph<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        self.connected_components()
            .into_values()
            .map(|component| {
                // Because indexing differs in the component subgraph, edges have to be
                // translated through the node index map built below.
                let edges: Vec<_> = self
                    .graph
                    .edge_references()
                    .filter(|edge| {
                        // One lookup would suffice because the components are disjoint.
                        component.contains(&edge.source()) && component.contains(&edge.target())
                    })
                    .collect();

                let mut graph = Graph::default();
                let mut map = HashMap::new();

                for index in &component {
                    let weight = self
                        .graph
                        .node_weight(*index)
                        .expect("clustering only contains indices of this graph");
                    let other_index = graph.add_node(weight.clone());
                    map.insert(*index, other_index);
                }

                for edge in edges {
                    let (a, b) = (edge.source(), edge.target());
                    graph.add_edge(map[&a], map[&b], edge.weight().clone());
                }

                TranslatedDiGraph {
                    graph: Self { graph },
                    map,
                }
            })
            .collect()
    }

    /// Splits the graph into the components described by `sets`.
    ///
    /// # Arguments
    ///
    /// * `sets` - The node partition defining the fragments.
    ///
    /// # Returns
    ///
    /// One fragment per set, together with the edges that were cut by the split.
    pub fn fragment_by(&self, sets: &DisjointSets<NodeIndex<usize>>) -> Vec<DiGraphFragment<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        sets.to_clustering()
            .into_iter()
            .map(|(repr, component)| {
                let mut graph = DiGraph::new();
                let mut cut_edges = Vec::new();

                // Insert all nodes into the new graph, recording the old indices.
                let map: HashMap<_, _> = component
                    .iter()
                    .map(|&node| {
                        let weight = self
                            .graph
                            .node_weight(node)
                            .cloned()
                            .expect("clustering only contains indices of this graph");
                        (node, graph.add_node(weight))
                    })
                    .collect();

                // Transfer edges into the subgraph, recording those leaving the fragment.
                for node in component {
                    for edge in self.graph.edges(node) {
                        let neighbor = edge.target();
                        let weight = edge.weight().clone();
                        let new_node = map[&node];

                        if sets.repr(neighbor) == repr {
                            graph.add_edge(new_node, map[&neighbor], weight);
                        } else {
                            cut_edges.push(CutEdge {
                                source: new_node,
                                weight,
                            });
                        }
                    }
                }

                DiGraphFragment {
                    fragment: graph,
                    cut_edges,
                }
            })
            .collect()
    }

    /// Creates a new graph containing `self` and `other` as separate components.
    ///
    /// # Arguments
    ///
    /// * `other` - The graph to append.
    ///
    /// # Returns
    ///
    /// The merged graph, able to translate indices valid for `other`.
    pub fn merge(&self, other: &Self) -> TranslatedDiGraph<N, E>
    where
        N: Clone,
        E: Clone,
    {
        let mut graph = self.clone();
        let mut map = HashMap::new();

        for (other_index, weight) in other.graph.node_references() {
            let index = graph.add_node(weight.clone());
            map.insert(other_index, index);
        }

        for edge in other.graph.edge_references() {
            let source = map[&edge.source()];
            let target = map[&edge.target()];
            graph.add_edge(source, target, edge.weight().clone());
        }

        TranslatedDiGraph { graph, map }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::graph::DiIntGraph;

    /// Builds `0 -> 1 -> 2 -> 3` plus the shortcut `0 -> 3`.
    fn diamond() -> DiIntGraph {
        let mut graph = DiIntGraph::new();
        let nodes: Vec<_> = (0..4).map(|w| graph.add_node(w)).collect();
        graph.add_edge(nodes[0], nodes[1], 0);
        graph.add_edge(nodes[1], nodes[2], 1);
        graph.add_edge(nodes[2], nodes[3], 2);
        graph.add_edge(nodes[0], nodes[3], 3);
        graph
    }

    /// Builds two disjoint edges: `0 -> 1` and `2 -> 3`.
    fn two_components() -> DiIntGraph {
        let mut graph = DiIntGraph::new();
        let nodes: Vec<_> = (0..4).map(|w| graph.add_node(w)).collect();
        graph.add_edge(nodes[0], nodes[1], 0);
        graph.add_edge(nodes[2], nodes[3], 1);
        graph
    }

    #[test]
    fn shortest_path_prefers_the_shortcut() {
        let graph = diamond();
        let path = graph
            .shortest_path(NodeIndex::new(0), NodeIndex::new(3))
            .unwrap();

        assert_eq!(path, vec![NodeIndex::new(0), NodeIndex::new(3)]);
    }

    #[test]
    fn shortest_path_returns_none_when_unreachable() {
        let graph = diamond();
        assert_eq!(
            graph.shortest_path(NodeIndex::new(3), NodeIndex::new(0)),
            None
        );
    }

    #[test]
    fn break_paths_disconnects_source_from_target() {
        let mut graph = diamond();
        let mut rng = StdRng::seed_from_u64(7);

        let cut = graph.break_paths(NodeIndex::new(0), NodeIndex::new(3), &mut rng);

        assert!(!cut.is_empty());
        assert_eq!(
            graph.shortest_path(NodeIndex::new(0), NodeIndex::new(3)),
            None
        );
    }

    #[test]
    fn break_paths_terminates_when_source_equals_target() {
        let mut graph = diamond();
        let mut rng = StdRng::seed_from_u64(7);

        // Regression: an empty sampling range used to panic / loop here.
        let cut = graph.break_paths(NodeIndex::new(0), NodeIndex::new(0), &mut rng);

        assert!(cut.is_empty());
        assert_eq!(graph.edge_count(), 4);
    }

    #[test]
    fn restore_edge_reattaches_a_cut_edge() {
        let mut graph = diamond();
        let mut rng = StdRng::seed_from_u64(1);
        let mut cut = graph.break_paths(NodeIndex::new(0), NodeIndex::new(3), &mut rng);
        let edges_after_cut = graph.edge_count();

        let cut_edge = cut.pop().unwrap();
        let source = cut_edge.source();
        graph.restore_edge(cut_edge, NodeIndex::new(1));

        assert_eq!(graph.edge_count(), edges_after_cut + 1);
        assert!(graph.contains_edge(source, NodeIndex::new(1)));
    }

    #[test]
    fn components_are_detected_regardless_of_direction() {
        let graph = two_components();

        assert_eq!(graph.num_components(), 2);
        assert_eq!(graph.connected_components().len(), 2);
    }

    #[test]
    fn connected_component_follows_edge_direction() {
        let graph = two_components();

        assert_eq!(graph.connected_component(NodeIndex::new(0)).len(), 2);
        assert_eq!(graph.connected_component(NodeIndex::new(1)).len(), 1);
    }

    #[test]
    fn overlap_is_empty_for_separate_components() {
        let graph = two_components();
        assert!(graph
            .overlap(NodeIndex::new(0), NodeIndex::new(2))
            .is_empty());
    }

    #[test]
    fn split_components_preserves_nodes_and_internal_edges() {
        let parts = two_components().split_components();

        assert_eq!(parts.len(), 2);
        for part in &parts {
            assert_eq!(part.graph().node_count(), 2);
            assert_eq!(part.graph().edge_count(), 1);
        }
    }

    #[test]
    fn split_components_translates_original_indices() {
        let graph = two_components();
        let parts = graph.split_components();

        let translated: Vec<_> = graph
            .node_indices()
            .map(|index| {
                parts
                    .iter()
                    .filter_map(|part| part.translate(index))
                    .count()
            })
            .collect();

        assert_eq!(translated, vec![1, 1, 1, 1], "each node lands in one part");
    }

    #[test]
    fn fragment_by_records_edges_leaving_a_fragment() {
        let graph = diamond();
        let mut sets = DisjointSets::new(graph.node_count());
        sets.union(NodeIndex::new(0), NodeIndex::new(1));
        sets.union(NodeIndex::new(2), NodeIndex::new(3));

        let fragments = graph.fragment_by(&sets);

        assert_eq!(fragments.len(), 2);
        let total_cut: usize = fragments.iter().map(|f| f.cut_edges.len()).sum();
        let total_internal: usize = fragments.iter().map(|f| f.fragment.edge_count()).sum();
        assert_eq!(total_cut, 2, "1 -> 2 and 0 -> 3 cross the split");
        assert_eq!(total_internal + total_cut, graph.edge_count());
    }

    #[test]
    fn merge_keeps_both_operands_as_separate_components() {
        let left = two_components();
        let right = diamond();

        let merged = left.merge(&right);

        assert_eq!(
            merged.graph().node_count(),
            left.node_count() + right.node_count()
        );
        assert_eq!(
            merged.graph().edge_count(),
            left.edge_count() + right.edge_count()
        );
        assert_eq!(merged.graph().num_components(), 3);
    }

    #[test]
    fn merge_translates_indices_of_the_appended_graph() {
        let left = two_components();
        let right = diamond();

        let merged = left.merge(&right);
        let translated = merged.translate(NodeIndex::new(0)).unwrap();

        assert_eq!(translated.index(), left.node_count());
    }
}
