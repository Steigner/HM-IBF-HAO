//! Recombine graphs.

use mahf::{
    components::recombination::{recombination, OptionalPair, Recombination},
    Component, ExecResult, Random, State,
};
use petgraph::graph::NodeIndex;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    graph::{algorithms, di::DiGraph, DiGraphFragment},
    problems::DirectedGraphProblem,
};

/// Applies a crossover to two parent graphs depending on crossover probability `pc`.
#[derive(Clone, Serialize, Deserialize)]
pub struct GraphPartitionCrossover {
    /// Crossover probability.
    pub pc: f64,
    /// If `false`, the second child is discarded.
    pub insert_both: bool,
}

impl GraphPartitionCrossover {
    pub fn from_params(pc: f64, insert_both: bool) -> Self {
        Self { pc, insert_both }
    }

    pub fn new<P>(pc: f64, insert_both: bool) -> Box<dyn Component<P>>
    where
        P: DirectedGraphProblem,
    {
        Box::new(Self::from_params(pc, insert_both))
    }

    /// Creates a new `GraphCrossover` which inserts only the first child.
    pub fn new_insert_single<P>(pc: f64) -> Box<dyn Component<P>>
    where
        P: DirectedGraphProblem,
    {
        Self::new(pc, false)
    }

    /// Creates a new `GraphCrossover` which inserts both children.
    pub fn new_insert_both<P>(pc: f64) -> Box<dyn Component<P>>
    where
        P: DirectedGraphProblem,
    {
        Self::new(pc, true)
    }
}

impl<P> Recombination<P> for GraphPartitionCrossover
where
    P: DirectedGraphProblem,
{
    fn recombine(
        &self,
        parent1: &P::Encoding,
        parent2: &P::Encoding,
        rng: &mut Random,
    ) -> OptionalPair<P::Encoding> {
        if rng.gen::<f64>() <= self.pc && !(parent1.node_count() < 3 || parent2.node_count() < 3) {
            let children = graph_crossover(parent1, parent2, rng);
            OptionalPair::from_pair(children, self.insert_both)
        } else {
            OptionalPair::None
        }
    }
}

impl<P> Component<P> for GraphPartitionCrossover
where
    P: DirectedGraphProblem,
{
    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        recombination(self, problem, state)
    }
}

pub fn generate_fragments<N, E>(
    graph: &DiGraph<N, E>,
    rng: &mut Random,
) -> [DiGraphFragment<N, E>; 2]
where
    N: Clone,
    E: Clone,
{
    let sets = algorithms::binomial_sets(graph, rng);
    graph
        .fragment_by(&sets)
        .try_into()
        .unwrap_or_else(|v: Vec<_>| panic!("Got {} instead of two fragments", v.len()))
}

pub fn merge_fragments<N, E>(
    a: DiGraphFragment<N, E>,
    b: DiGraphFragment<N, E>,
    rng: &mut Random,
) -> DiGraph<N, E>
where
    N: Clone,
    E: Clone,
{
    let DiGraphFragment {
        cut_edges: cut_edges_a,
        fragment: frag_a,
    } = a;
    let DiGraphFragment {
        cut_edges: mut cut_edges_b,
        fragment: frag_b,
    } = b;

    let component = frag_a.merge(&frag_b);

    // Translate cut edges.
    for edge in &mut cut_edges_b {
        edge.map_source(|source| component.translate(source).unwrap());
    }

    let mut graph = component.into_graph();

    // Insert cut edges from `a` into `b` and the other way around.
    let num_nodes_a = frag_a.node_count();
    let num_nodes_b = frag_b.node_count();

    let range_a = 0..num_nodes_a;
    let range_b = num_nodes_a..num_nodes_a + num_nodes_b;

    for edge in cut_edges_a {
        let target = NodeIndex::new(rng.gen_range(range_b.clone()));
        graph.restore_edge(edge, target);
    }

    for edge in cut_edges_b {
        let target = NodeIndex::new(rng.gen_range(range_a.clone()));
        graph.restore_edge(edge, target);
    }

    graph
}

pub fn graph_crossover<N, E>(
    parent1: &DiGraph<N, E>,
    parent2: &DiGraph<N, E>,
    rng: &mut Random,
) -> [DiGraph<N, E>; 2]
where
    N: Clone + std::fmt::Display,
    E: Clone + std::fmt::Display,
{
    let [a1, b1] = generate_fragments(parent1, rng);
    let [a2, b2] = generate_fragments(parent2, rng);

    let child1 = merge_fragments(a1, a2, rng);
    let child2 = merge_fragments(b1, b2, rng);

    [child1, child2]
}
