//! Mutate graphs.

use mahf::{population::AsSolutionsMut, Component, ExecResult, State};
use rand::{seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};

use crate::problems::DirectedGraphProblem;

/// Inserts and removes nodes.
#[derive(Clone, Serialize, Deserialize)]
pub struct NodeInsertion {
    /// Probability of inserting or removing a single node.
    pub rm: f64,
}

impl NodeInsertion {
    pub fn from_params(rm: f64) -> Self {
        Self { rm }
    }

    pub fn new<P>(rm: f64) -> Box<dyn Component<P>>
    where
        P: DirectedGraphProblem,
    {
        Box::new(Self::from_params(rm))
    }
}

impl<P> Component<P> for NodeInsertion
where
    P: DirectedGraphProblem,
{
    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        let mut populations = state.populations_mut();
        let mut rng = state.random_mut();

        let node_types = problem.node_types();
        let edge_types = problem.edge_types();

        for solution in populations.current_mut().as_solutions_mut() {
            if rng.gen_bool(self.rm) {
                if solution.node_count() < 2 || rng.gen_bool(0.5) {
                    // Insert new node.
                    let nodes: Vec<_> = solution.node_indices().collect();
                    let new_node = solution.gen_node(&node_types, &mut *rng);

                    // Connect with original nodes forward and backward.
                    for node in nodes {
                        if rng.gen_bool(0.5) {
                            solution.gen_edge(new_node, node, &edge_types, &mut *rng);
                        }
                        if rng.gen_bool(0.5) {
                            solution.gen_edge(node, new_node, &edge_types, &mut *rng);
                        }
                    }
                } else {
                    // Remove random node.
                    let index = solution.gen_index(&mut *rng);
                    solution.remove_node(index);
                }
            }
        }

        Ok(())
    }
}

/// Mutates node weights.
#[derive(Clone, Serialize, Deserialize)]
pub struct NodeWeightMutation {
    /// Probability of resampling the node weight.
    pub rm: f64,
}

impl NodeWeightMutation {
    pub fn from_params(rm: f64) -> Self {
        Self { rm }
    }

    pub fn new<P>(rm: f64) -> Box<dyn Component<P>>
    where
        P: DirectedGraphProblem,
    {
        Box::new(Self::from_params(rm))
    }
}

impl<P> Component<P> for NodeWeightMutation
where
    P: DirectedGraphProblem,
{
    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        let mut populations = state.populations_mut();
        let mut rng = state.random_mut();

        let node_types = problem.node_types();

        for solution in populations.current_mut().as_solutions_mut() {
            for node in solution.node_indices() {
                if rng.gen_bool(self.rm) {
                    solution.gen_node_weight(node, &node_types, &mut *rng);
                }
            }
        }

        Ok(())
    }
}

/// Inserts and removes edges.
#[derive(Clone, Serialize, Deserialize)]
pub struct EdgeInsertion {
    /// Probability of inserting or removing any edges.
    pub rm: f64,
}

impl EdgeInsertion {
    pub fn from_params(rm: f64) -> Self {
        Self { rm }
    }

    pub fn new<P>(rm: f64) -> Box<dyn Component<P>>
    where
        P: DirectedGraphProblem,
    {
        Box::new(Self::from_params(rm))
    }
}

impl<P> Component<P> for EdgeInsertion
where
    P: DirectedGraphProblem,
{
    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        let mut populations = state.populations_mut();
        let mut rng = state.random_mut();

        let edge_types = problem.edge_types();

        for solution in populations.current_mut().as_solutions_mut() {
            if rng.gen_bool(0.5) {
                // Insert random missing edges.
                let mut nodes_i: Vec<_> = solution.node_indices().collect();
                nodes_i.shuffle(&mut *rng);
                let mut nodes_j: Vec<_> = solution.node_indices().collect();
                nodes_j.shuffle(&mut *rng);

                for i in nodes_i {
                    for j in nodes_j.clone() {
                        if i == j || solution.contains_edge(i, j) {
                            continue;
                        }

                        if rng.gen_bool(self.rm) {
                            solution.gen_edge(i, j, &edge_types, &mut *rng);
                        }
                    }
                }
            } else {
                // Remove random existing edges.
                for edge in solution.edge_indices() {
                    if rng.gen_bool(self.rm) {
                        solution.remove_edge(edge);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Mutates edge weights.
#[derive(Clone, Serialize, Deserialize)]
pub struct EdgeWeightMutation {
    pub rm: f64,
}

impl EdgeWeightMutation {
    pub fn from_params(rm: f64) -> Self {
        Self { rm }
    }

    pub fn new<P>(rm: f64) -> Box<dyn Component<P>>
    where
        P: DirectedGraphProblem,
    {
        Box::new(Self::from_params(rm))
    }
}

impl<P> Component<P> for EdgeWeightMutation
where
    P: DirectedGraphProblem,
{
    fn execute(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        let mut populations = state.populations_mut();
        let mut rng = state.random_mut();

        let edge_types = problem.edge_types();

        for solution in populations.current_mut().as_solutions_mut() {
            for edge in solution.edge_indices() {
                if rng.gen_bool(self.rm) {
                    solution.gen_edge_weight(edge, &edge_types, &mut *rng);
                }
            }
        }

        Ok(())
    }
}
