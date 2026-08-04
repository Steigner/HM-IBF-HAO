//! Partitioning algorithms that split a graph's nodes into disjoint sets.

use petgraph::graph::NodeIndex;
use rand::{distributions::Distribution, seq::SliceRandom, RngCore};
use rand_distr::Binomial;

use crate::graph::{set::DisjointSets, DiGraph};

/// Splits the nodes into exactly two sets with a binomially distributed size ratio.
///
/// # Arguments
///
/// * `graph` - The graph whose nodes are partitioned.
/// * `rng` - Random number generator used for the split.
///
/// # Returns
///
/// A disjoint-set structure containing exactly two sets.
///
/// # Panics
///
/// Panics if the graph has fewer than two nodes.
pub fn binomial_sets<N, E, R>(graph: &DiGraph<N, E>, rng: &mut R) -> DisjointSets<NodeIndex<usize>>
where
    R: RngCore + ?Sized,
{
    let n = graph.node_count();
    assert!(
        n >= 2,
        "a binomial split requires at least 2 nodes, but the graph has {n}"
    );

    // `n - 2` trials plus the mandatory first node keep the split size in `1..=n - 1`,
    // so both sets are guaranteed to be non-empty.
    let dist = Binomial::new(n as u64 - 2, 0.5).expect("0.5 is a valid success probability");
    let split = dist.sample(rng) as usize + 1;

    let mut nodes: Vec<_> = graph.node_indices().collect();
    nodes.shuffle(rng);

    let mut sets = DisjointSets::new(n);

    for node in &nodes[1..split] {
        sets.union(nodes[0], *node);
    }

    for node in &nodes[split + 1..] {
        sets.union(nodes[split], *node);
    }

    sets
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::graph::DiIntGraph;

    /// Builds the path `0 -> 1 -> ... -> n-1`.
    fn path(n: usize) -> DiIntGraph {
        let mut graph = DiIntGraph::new();
        let nodes: Vec<_> = (0..n).map(|w| graph.add_node(w as u32)).collect();
        for pair in nodes.windows(2) {
            graph.add_edge(pair[0], pair[1], 0);
        }
        graph
    }

    #[test]
    fn binomial_sets_always_produces_two_non_empty_sets() {
        let graph = path(6);
        for seed in 0..32 {
            let mut rng = StdRng::seed_from_u64(seed);
            let sets = binomial_sets(&graph, &mut rng);

            assert_eq!(sets.count(), 2, "seed {seed}");
            assert!(sets.counts().values().all(|&count| count > 0));
            assert_eq!(sets.counts().values().sum::<usize>(), graph.node_count());
        }
    }

    #[test]
    fn binomial_sets_handles_the_smallest_valid_graph() {
        let graph = path(2);
        let mut rng = StdRng::seed_from_u64(0);

        assert_eq!(binomial_sets(&graph, &mut rng).count(), 2);
    }

    #[test]
    fn binomial_sets_covers_every_node() {
        let graph = path(9);
        let mut rng = StdRng::seed_from_u64(11);

        let sets = binomial_sets(&graph, &mut rng);

        assert_eq!(sets.counts().values().sum::<usize>(), graph.node_count());
    }

    #[test]
    #[should_panic(expected = "at least 2 nodes")]
    fn binomial_sets_rejects_graphs_that_cannot_be_split() {
        let graph = path(1);
        binomial_sets(&graph, &mut StdRng::seed_from_u64(0));
    }
}
