//! Randomized construction of [`DiGraph`] nodes, edges and standard topologies.

use itertools::Itertools;
use petgraph::graph::{EdgeIndex, NodeIndex};
use rand::{seq::SliceRandom, Rng};

use crate::{graph::di::DiGraph, utils::gen_distinct};

impl<N, E> DiGraph<N, E> {
    /// Generates a random node index that is valid for this graph.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator used for the draw.
    ///
    /// # Returns
    ///
    /// A uniformly sampled node index.
    ///
    /// # Panics
    ///
    /// Panics if the graph has no nodes.
    pub fn gen_index<R>(&self, rng: &mut R) -> NodeIndex<usize>
    where
        R: Rng + ?Sized,
    {
        assert!(
            self.node_count() > 0,
            "cannot sample a node index from an empty graph"
        );
        NodeIndex::new(rng.gen_range(0..self.node_count()))
    }

    /// Generates a random node index distinct from all of `indices`.
    ///
    /// # Arguments
    ///
    /// * `indices` - The node indices to exclude.
    /// * `rng` - Random number generator used for the draw.
    ///
    /// # Returns
    ///
    /// A uniformly sampled node index, or `None` if every node is excluded.
    pub fn gen_distinct_index<I, R>(&self, indices: I, rng: &mut R) -> Option<NodeIndex<usize>>
    where
        R: Rng + ?Sized,
        I: IntoIterator<Item = NodeIndex<usize>>,
    {
        gen_distinct(
            0..self.node_count(),
            indices.into_iter().map(|i| i.index()),
            rng,
        )
        .map(NodeIndex::new)
    }

    /// Adds an unconnected node with a weight sampled from `weights`.
    ///
    /// # Arguments
    ///
    /// * `weights` - The candidate node weights; must be non-empty.
    /// * `rng` - Random number generator used for the draw.
    ///
    /// # Returns
    ///
    /// The index of the new node.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is empty.
    pub fn gen_node<R>(&mut self, weights: &[N], rng: &mut R) -> NodeIndex<usize>
    where
        R: Rng + ?Sized,
        N: Clone,
    {
        let weight = weights
            .choose(rng)
            .cloned()
            .expect("node weights must not be empty");
        self.add_node(weight)
    }

    /// Adds an edge between `a` and `b` with a weight sampled from `weights`.
    ///
    /// # Arguments
    ///
    /// * `a` - The source node.
    /// * `b` - The target node.
    /// * `weights` - The candidate edge weights; must be non-empty.
    /// * `rng` - Random number generator used for the draw.
    ///
    /// # Returns
    ///
    /// The index of the new edge.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is empty.
    pub fn gen_edge<R>(
        &mut self,
        a: NodeIndex<usize>,
        b: NodeIndex<usize>,
        weights: &[E],
        rng: &mut R,
    ) -> EdgeIndex<usize>
    where
        R: Rng + ?Sized,
        E: Clone,
    {
        let weight = weights
            .choose(rng)
            .cloned()
            .expect("edge weights must not be empty");
        self.add_edge(a, b, weight)
    }

    /// Replaces the weight of the node at `index` with a sample from `weights`.
    ///
    /// Does nothing if the node does not exist.
    ///
    /// # Arguments
    ///
    /// * `index` - The node to mutate.
    /// * `weights` - The candidate node weights; must be non-empty.
    /// * `rng` - Random number generator used for the draw.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is empty.
    pub fn gen_node_weight<R>(&mut self, index: NodeIndex<usize>, weights: &[N], rng: &mut R)
    where
        R: Rng + ?Sized,
        N: Clone,
    {
        let new_weight = weights
            .choose(rng)
            .cloned()
            .expect("node weights must not be empty");
        if let Some(weight) = self.node_weight_mut(index) {
            *weight = new_weight;
        }
    }

    /// Replaces the weight of the edge at `index` with a sample from `weights`.
    ///
    /// Does nothing if the edge does not exist.
    ///
    /// # Arguments
    ///
    /// * `index` - The edge to mutate.
    /// * `weights` - The candidate edge weights; must be non-empty.
    /// * `rng` - Random number generator used for the draw.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is empty.
    pub fn gen_edge_weight<R>(&mut self, index: EdgeIndex<usize>, weights: &[E], rng: &mut R)
    where
        R: Rng + ?Sized,
        E: Clone,
    {
        let new_weight = weights
            .choose(rng)
            .cloned()
            .expect("edge weights must not be empty");
        if let Some(weight) = self.edge_weight_mut(index) {
            *weight = new_weight;
        }
    }

    /// Generates the singleton graph with one randomly weighted node.
    ///
    /// # Arguments
    ///
    /// * `weights` - The candidate node weights; must be non-empty.
    /// * `rng` - Random number generator used for the draw.
    ///
    /// # Returns
    ///
    /// The singleton graph.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is empty.
    pub fn gen_singleton<R>(weights: &[N], rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        N: Clone,
    {
        let mut graph = Self::new();
        graph.gen_node(weights, rng);
        graph
    }

    /// Generates the star graph with `n` outer nodes and randomly sampled weights.
    ///
    /// All edges are directed towards the center node.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of outer nodes.
    /// * `node_weights` - The candidate node weights; must be non-empty.
    /// * `edge_weights` - The candidate edge weights; must be non-empty.
    /// * `rng` - Random number generator used for the draws.
    ///
    /// # Returns
    ///
    /// The star graph with `n + 1` nodes.
    ///
    /// # Panics
    ///
    /// Panics if either weight slice is empty.
    pub fn gen_star<R>(n: usize, node_weights: &[N], edge_weights: &[E], rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        N: Clone,
        E: Clone,
    {
        let mut graph = Self::new();
        let center = graph.gen_node(node_weights, rng);

        for _ in 0..n {
            let outer = graph.gen_node(node_weights, rng);
            graph.gen_edge(outer, center, edge_weights, rng);
        }

        graph
    }

    /// Generates the wheel graph with `n` outer nodes and randomly sampled weights.
    ///
    /// Edges are directed towards the center node and in one direction around the ring.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of outer nodes.
    /// * `node_weights` - The candidate node weights; must be non-empty.
    /// * `edge_weights` - The candidate edge weights; must be non-empty.
    /// * `rng` - Random number generator used for the draws.
    ///
    /// # Returns
    ///
    /// The wheel graph with `n + 1` nodes.
    ///
    /// # Panics
    ///
    /// Panics if either weight slice is empty.
    pub fn gen_wheel<R>(n: usize, node_weights: &[N], edge_weights: &[E], rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        N: Clone,
        E: Clone,
    {
        let mut graph = Self::new();
        let center = graph.gen_node(node_weights, rng);

        let mut outer_nodes = Vec::with_capacity(n);
        for _ in 0..n {
            let outer = graph.gen_node(node_weights, rng);
            graph.gen_edge(outer, center, edge_weights, rng);
            outer_nodes.push(outer);
        }

        for (a, b) in outer_nodes.into_iter().circular_tuple_windows() {
            graph.gen_edge(a, b, edge_weights, rng);
        }

        graph
    }

    /// Generates the complete graph with `n` nodes and randomly sampled weights.
    ///
    /// Each ordered node pair is visited once and connected in a randomly chosen direction.
    ///
    /// Because both `(a, b)` and `(b, a)` are visited, a pair can end up with edges in both
    /// directions; the resulting edge count therefore lies in `n * (n - 1) / 2 ..= n * (n - 1)`.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of nodes.
    /// * `node_weights` - The candidate node weights; must be non-empty.
    /// * `edge_weights` - The candidate edge weights; must be non-empty.
    /// * `rng` - Random number generator used for the draws.
    ///
    /// # Returns
    ///
    /// The generated graph.
    ///
    /// # Panics
    ///
    /// Panics if either weight slice is empty.
    pub fn gen_complete<R>(n: usize, node_weights: &[N], edge_weights: &[E], rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        N: Clone,
        E: Clone,
    {
        let mut graph = Self::new();

        let outer_nodes: Vec<_> = (0..n).map(|_| graph.gen_node(node_weights, rng)).collect();

        for a in &outer_nodes {
            for b in &outer_nodes {
                if a == b {
                    continue;
                }

                if rng.gen_bool(0.5) {
                    graph.gen_edge(*a, *b, edge_weights, rng);
                } else {
                    graph.gen_edge(*b, *a, edge_weights, rng);
                }
            }
        }

        graph
    }

    /// Generates the ring graph with `n` nodes and randomly sampled weights.
    ///
    /// The edges are directed in one direction around the ring.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of nodes.
    /// * `node_weights` - The candidate node weights; must be non-empty.
    /// * `edge_weights` - The candidate edge weights; must be non-empty.
    /// * `rng` - Random number generator used for the draws.
    ///
    /// # Returns
    ///
    /// The ring graph.
    ///
    /// # Panics
    ///
    /// Panics if either weight slice is empty.
    pub fn gen_ring<R>(n: usize, node_weights: &[N], edge_weights: &[E], rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        N: Clone,
        E: Clone,
    {
        let mut graph = Self::new();

        let nodes: Vec<_> = (0..n).map(|_| graph.gen_node(node_weights, rng)).collect();

        for (a, b) in nodes.into_iter().circular_tuple_windows() {
            graph.gen_edge(a, b, edge_weights, rng);
        }

        graph
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::graph::DiIntGraph;

    const NODE_WEIGHTS: [u32; 3] = [0, 1, 2];
    const EDGE_WEIGHTS: [u32; 2] = [0, 1];

    fn rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn gen_singleton_has_exactly_one_node_and_no_edges() {
        let graph = DiIntGraph::gen_singleton(&NODE_WEIGHTS, &mut rng());
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn gen_star_points_every_outer_node_at_the_center() {
        let graph = DiIntGraph::gen_star(4, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng());

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4);
        let center = NodeIndex::new(0);
        for outer in 1..5 {
            assert!(graph.contains_edge(NodeIndex::new(outer), center));
        }
    }

    #[test]
    fn gen_wheel_adds_a_ring_on_top_of_the_star() {
        let graph = DiIntGraph::gen_wheel(4, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng());

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 8, "4 spokes plus a 4-cycle");
    }

    #[test]
    fn gen_ring_produces_a_single_cycle() {
        let graph = DiIntGraph::gen_ring(5, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng());

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 5);
        assert_eq!(graph.num_components(), 1);
    }

    #[test]
    fn gen_complete_connects_every_pair_in_at_least_one_direction() {
        let n = 5;
        let graph = DiIntGraph::gen_complete(n, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng());

        assert_eq!(graph.node_count(), n);
        // Every ordered pair is visited, so a pair may receive both orientations.
        assert!((n * (n - 1) / 2..=n * (n - 1)).contains(&graph.edge_count()));
        for a in 0..n {
            for b in (a + 1)..n {
                let (a, b) = (NodeIndex::new(a), NodeIndex::new(b));
                assert!(
                    graph.contains_edge(a, b) || graph.contains_edge(b, a),
                    "every pair must be connected"
                );
            }
        }
    }

    #[test]
    fn generated_weights_come_from_the_candidate_slices() {
        let graph = DiIntGraph::gen_wheel(6, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng());

        assert!(graph.node_weights().all(|w| NODE_WEIGHTS.contains(w)));
        assert!(graph.edge_weights().all(|w| EDGE_WEIGHTS.contains(w)));
    }

    #[test]
    fn gen_distinct_index_excludes_the_given_indices() {
        let graph = DiIntGraph::gen_ring(3, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng());
        let excluded = [NodeIndex::new(0), NodeIndex::new(1)];

        let index = graph.gen_distinct_index(excluded, &mut rng()).unwrap();

        assert_eq!(index, NodeIndex::new(2));
    }

    #[test]
    fn gen_distinct_index_returns_none_when_all_nodes_are_excluded() {
        let graph = DiIntGraph::gen_ring(3, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng());
        let excluded: Vec<_> = graph.node_indices().collect();

        assert_eq!(graph.gen_distinct_index(excluded, &mut rng()), None);
    }

    #[test]
    fn gen_node_weight_replaces_the_weight_in_place() {
        let mut graph = DiIntGraph::gen_ring(3, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng());
        let index = NodeIndex::new(1);

        graph.gen_node_weight(index, &[7], &mut rng());

        assert_eq!(graph.node_weight(index), Some(&7));
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn gen_edge_weight_on_a_missing_edge_is_a_noop() {
        let mut graph = DiIntGraph::gen_ring(3, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng());

        graph.gen_edge_weight(EdgeIndex::new(99), &[7], &mut rng());

        assert!(graph.edge_weights().all(|w| EDGE_WEIGHTS.contains(w)));
    }
}
