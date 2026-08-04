//! Local Search (LS) island builder.

use grahf::problems::algorithm_design::builder::MetaheuristicIslandBuilder;
use irace_rs::param_space::ParamSpace;
use mahf::{params::Params, prelude::*};

use crate::{
    islands::{default_initialization, make_mutation, safe_boundary, ResetIterations},
    problems::{DimensionAwareDomain, RealValuedProblem},
};

/// Island builder for Local Search using neighborhood mutation.
#[derive(Clone)]
pub struct Builder {
    /// Allowed island working dimensions in offsets along the backbone.
    pub dimensions_allowed: Vec<u32>,
    /// Maximum number of iterations IRACE can assign to this island.
    pub max_iterations: u32,
    /// Maximum number of neighbors generated per step.
    pub max_neighbors: u32,
}

impl Builder {
    /// Creates a new LS island builder.
    pub fn new<P: RealValuedProblem + DimensionAwareDomain>(
        dimensions_allowed: &[u32],
        max_iterations: u32,
        max_neighbors: u32,
    ) -> Box<dyn MetaheuristicIslandBuilder<P>> {
        Box::new(Self {
            dimensions_allowed: dimensions_allowed.to_vec(),
            max_iterations,
            max_neighbors,
        })
    }
}

impl<P: RealValuedProblem + DimensionAwareDomain> MetaheuristicIslandBuilder<P> for Builder {
    fn id(&self) -> String {
        "ls".to_string()
    }

    fn build_init(&self, params: Params) -> ExecResult<Box<dyn Component<P>>> {
        default_initialization(params)
    }

    fn build_island(&self, mut params: Params) -> ExecResult<Box<dyn Component<P>>> {
        let iterations = params.try_extract::<u32>("iterations")?;
        let num_neighbors = params.try_extract::<u32>("num_neighbors")?;

        let neighbors = make_mutation(
            &params.try_extract::<String>("mutation")?,
            params.try_extract::<f64>("mutation_strength")?,
        )?;

        Ok(Configuration::builder()
            .do_(ResetIterations::new())
            .while_(conditions::LessThanN::iterations(iterations), |builder| {
                builder
                    .do_(selection::CloneSingle::new(num_neighbors))
                    .do_(neighbors)
                    .do_(safe_boundary::IslandDimensionSaturation::new())
                    .evaluate()
                    .update_best_individual()
                    .do_(replacement::MuPlusLambda::new(1))
                    .do_(logging::Logger::new())
            })
            .build_component())
    }

    fn param_space(&self) -> ParamSpace {
        ParamSpace::new()
            .with_integer("iterations", 1, self.max_iterations, false)
            .with_integer("num_neighbors", 4, self.max_neighbors, false)
            .with_categorical("population_size", [1u32])
            .with_real("mutation_strength", 1e-3, 0.5, true)
            .with_categorical_names("mutation", ["normal", "uniform"])
            .with_categorical("dimension", self.dimensions_allowed.clone())
    }
}
