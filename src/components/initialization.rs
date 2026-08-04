//! Initialize graphs in the search space.

use eyre::ensure;
use mahf::{
    components::initialization::{initialization, Initialization},
    Component, ExecResult, Random, State,
};
use rand::Rng;
use rand_distr::Binomial;
use serde::{Deserialize, Serialize};

use crate::{graph::DiIntGraph, problems::DirectedGraphProblem};

/// Adds `|E| ~ Binomial(n^2, p)` edges with no loops to the graph.
pub fn add_edges(graph: &mut DiIntGraph, p: f64, weights: &[u32], rng: &mut Random) {
    for i in graph.node_indices() {
        for j in graph.node_indices() {
            if i == j || rng.gen_bool(1.0 - p) {
                continue;
            }

            graph.gen_edge(i, j, weights, rng);
        }
    }
}

/// Generates random graphs within the boundaries of the search space.
///
/// The generated graphs have `|N| ~ Uniform(1, n)` nodes and `|E| ~ Binomial(|N|^2, p)` edges with no loops.
///
/// The node and edge types are uniformly distributed.
#[derive(Clone, Serialize, Deserialize)]
pub struct UniformRandomGraph {
    /// The maximal number of nodes in the graph.
    pub n: u32,
    /// The probability of an edge existing.
    pub p: f64,
    /// Size of the population to be generated.
    pub population_size: u32,
}

impl UniformRandomGraph {
    pub fn from_params(n: u32, p: f64, population_size: u32) -> ExecResult<Self> {
        ensure!(
            n > 1,
            "the graph needs to be initialized with at least two nodes"
        );
        ensure!(
            (0.0..=1.0).contains(&p),
            "the edge probability must be between 0 and 1"
        );
        Ok(Self {
            n,
            p,
            population_size,
        })
    }

    pub fn new<P>(n: u32, p: f64, population_size: u32) -> ExecResult<Box<dyn Component<P>>>
    where
        P: DirectedGraphProblem,
    {
        Ok(Box::new(Self::from_params(n, p, population_size)?))
    }
}

impl<P> Initialization<P> for UniformRandomGraph
where
    P: DirectedGraphProblem,
{
    fn initialize(&self, problem: &P, rng: &mut Random) -> Vec<P::Encoding> {
        let node_types = problem.node_types();
        let edge_types = problem.edge_types();

        (0..self.population_size)
            .map(|_| {
                let mut graph = DiIntGraph::new();

                let num_nodes = rng.gen_range(1..self.n) as usize;
                for _ in 0..num_nodes {
                    graph.gen_node(&node_types, rng);
                }

                add_edges(&mut graph, self.p, &edge_types, rng);

                graph
            })
            .collect()
    }
}

impl<P> Component<P> for UniformRandomGraph
where
    P: DirectedGraphProblem,
{
    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        initialization(self, problem, state)
    }
}

/// Generates random graphs within the boundaries of the search space.
///
/// The generated graphs have `|N| ~ Binomial(2n, 0.5) - n` nodes and `|E| ~ Binomial(|N|^2, 0.5)` edges with no loops.
///
/// The node and edge types are uniformly distributed.
#[derive(Clone, Serialize, Deserialize)]
pub struct BinomialRandomGraph {
    /// The mean number of nodes in the graph.
    pub n: u32,
    /// The probability of an edge existing.
    pub p: f64,
    /// Size of the population to be generated.
    pub population_size: u32,
}

impl BinomialRandomGraph {
    pub fn from_params(n: u32, p: f64, population_size: u32) -> ExecResult<Self> {
        ensure!(
            n > 1,
            "the graph needs to be initialized with at least two nodes"
        );
        ensure!(
            (0.0..=1.0).contains(&p),
            "the edge probability must be between 0 and 1"
        );
        Ok(Self {
            n,
            p,
            population_size,
        })
    }

    pub fn new<P>(n: u32, p: f64, population_size: u32) -> ExecResult<Box<dyn Component<P>>>
    where
        P: DirectedGraphProblem,
    {
        Ok(Box::new(Self::from_params(n, p, population_size)?))
    }
}

impl<P> Initialization<P> for BinomialRandomGraph
where
    P: DirectedGraphProblem,
{
    fn initialize(&self, problem: &P, rng: &mut Random) -> Vec<P::Encoding> {
        let node_types = problem.node_types();
        let edge_types = problem.edge_types();

        let node_distribution = Binomial::new(self.n as u64 / 2, 0.5).unwrap();
        (0..self.population_size)
            .map(|_| {
                let mut graph = DiIntGraph::new();

                let num_nodes = rng.sample(node_distribution);
                for _ in 0..num_nodes {
                    graph.gen_node(&node_types, rng);
                }

                add_edges(&mut graph, self.p, &edge_types, rng);

                graph
            })
            .collect()
    }
}

impl<P> Component<P> for BinomialRandomGraph
where
    P: DirectedGraphProblem,
{
    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        initialization(self, problem, state)
    }
}
