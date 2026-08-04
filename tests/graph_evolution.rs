//! Integration tests for the graph search space that GRAHF evolves over.
//!
//! These exercise the crate through its public API only: generate graphs, fragment them,
//! recombine them and check that the results stay valid island-graph encodings.

use grahf::{
    components::recombination::{generate_fragments, graph_crossover, merge_fragments},
    graph::DiIntGraph,
};
use mahf::Random;

const NODE_WEIGHTS: [u32; 4] = [0, 1, 2, 3];
const EDGE_WEIGHTS: [u32; 3] = [0, 1, 2];

/// Builds a deterministic random generator for the given seed.
fn rng(seed: u64) -> Random {
    Random::new(seed)
}

#[test]
fn generated_topologies_only_use_the_offered_weights() {
    let mut rng = rng(1);

    for n in 3..8 {
        for graph in [
            DiIntGraph::gen_star(n, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng),
            DiIntGraph::gen_wheel(n, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng),
            DiIntGraph::gen_complete(n, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng),
            DiIntGraph::gen_ring(n, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng),
        ] {
            assert!(graph.node_weights().all(|w| NODE_WEIGHTS.contains(w)));
            assert!(graph.edge_weights().all(|w| EDGE_WEIGHTS.contains(w)));
        }
    }
}

#[test]
fn fragmenting_and_merging_preserves_the_node_count() {
    let mut rng = rng(2);

    for seed_offset in 0..10 {
        let graph = DiIntGraph::gen_wheel(6, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng);
        let mut fragment_rng = Random::new(100 + seed_offset);
        let [a, b] = generate_fragments(&graph, &mut fragment_rng);

        let merged = merge_fragments(a, b, &mut rng);

        assert_eq!(
            merged.node_count(),
            graph.node_count(),
            "seed {seed_offset}"
        );
    }
}

#[test]
fn fragmenting_splits_a_graph_into_exactly_two_non_empty_parts() {
    let mut rng = rng(3);
    let graph = DiIntGraph::gen_complete(7, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng);

    let [a, b] = generate_fragments(&graph, &mut rng);

    assert!(a.fragment.node_count() > 0);
    assert!(b.fragment.node_count() > 0);
    assert_eq!(
        a.fragment.node_count() + b.fragment.node_count(),
        graph.node_count()
    );
}

#[test]
fn crossover_produces_two_children_of_the_combined_parent_sizes() {
    let mut rng = rng(4);
    let parent1 = DiIntGraph::gen_star(5, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng);
    let parent2 = DiIntGraph::gen_wheel(6, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng);

    let [child1, child2] = graph_crossover(&parent1, &parent2, &mut rng);

    // Each child takes one fragment from each parent, so together the children hold every
    // node of both parents.
    assert_eq!(
        child1.node_count() + child2.node_count(),
        parent1.node_count() + parent2.node_count()
    );
}

#[test]
fn crossover_children_keep_valid_node_and_edge_weights() {
    let mut rng = rng(5);
    let parent1 = DiIntGraph::gen_complete(5, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng);
    let parent2 = DiIntGraph::gen_wheel(5, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng);

    for child in graph_crossover(&parent1, &parent2, &mut rng) {
        assert!(child.node_weights().all(|w| NODE_WEIGHTS.contains(w)));
        assert!(child.edge_weights().all(|w| EDGE_WEIGHTS.contains(w)));
    }
}

#[test]
fn crossover_is_reproducible_for_the_same_seed() {
    let build = || {
        let mut rng = rng(6);
        let parent1 = DiIntGraph::gen_star(5, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng);
        let parent2 = DiIntGraph::gen_wheel(5, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng);
        graph_crossover(&parent1, &parent2, &mut rng)
    };

    let [first_a, first_b] = build();
    let [second_a, second_b] = build();

    assert_eq!(first_a, second_a);
    assert_eq!(first_b, second_b);
}

#[test]
fn fragment_and_merge_round_trip_keeps_the_graph_connected() {
    let mut rng = rng(8);
    let graph = DiIntGraph::gen_ring(6, &NODE_WEIGHTS, &EDGE_WEIGHTS, &mut rng);
    let [a, b] = generate_fragments(&graph, &mut rng);

    let merged = merge_fragments(a, b, &mut rng);

    // Every cut edge is restored into the opposite fragment, so no node is left isolated.
    assert_eq!(merged.num_components(), 1, "{merged}");
}
