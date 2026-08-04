//! Random Search (RS) island builder.

use grahf::problems::algorithm_design::builder::MetaheuristicIslandBuilder;
use irace_rs::param_space::ParamSpace;
use mahf::{params::Params, prelude::*};

use crate::{
    islands::{default_initialization, safe_boundary, safe_diversity, ResetIterations},
    problems::{DimensionAwareDomain, RealValuedProblem},
};

/// Island builder for Random Search via partial random spread mutation.
#[derive(Clone)]
pub struct Builder {
    /// Allowed island working dimensions in offsets along the backbone.
    pub dimensions_allowed: Vec<u32>,
    /// Maximum number of iterations IRACE can assign to this island.
    pub max_iterations: u32,
    /// Maximum population size IRACE can assign to this island.
    pub max_population_size: u32,
}

impl Builder {
    /// Creates a new RS island builder.
    pub fn new<P: RealValuedProblem + DimensionAwareDomain>(
        dimensions_allowed: &[u32],
        max_iterations: u32,
        max_population_size: u32,
    ) -> Box<dyn MetaheuristicIslandBuilder<P>> {
        Box::new(Self {
            dimensions_allowed: dimensions_allowed.to_vec(),
            max_iterations,
            max_population_size,
        })
    }
}

impl<P: RealValuedProblem + DimensionAwareDomain> MetaheuristicIslandBuilder<P> for Builder {
    fn id(&self) -> String {
        "rs".to_string()
    }

    fn build_init(&self, params: Params) -> ExecResult<Box<dyn Component<P>>> {
        default_initialization(params)
    }

    fn build_island(&self, mut params: Params) -> ExecResult<Box<dyn Component<P>>> {
        let iterations = params.try_extract::<u32>("iterations")?;
        let mutation_strength = params.try_extract::<f64>("mutation_strength")?;

        Ok(Configuration::builder()
            .do_(ResetIterations::new())
            .while_(conditions::LessThanN::iterations(iterations), |builder| {
                builder
                    // SafePartialRandomSpread samples from domain_for_dimension(D) so
                    // values stay within the per-position bounds of the active dimension.
                    .do_(safe_boundary::SafePartialRandomSpread::new(
                        mutation_strength,
                    ))
                    .evaluate()
                    .update_best_individual()
                    .do_(safe_diversity::SafePairwiseDistanceDiversity::new())
                    .do_(logging::Logger::new())
            })
            .build_component())
    }

    fn param_space(&self) -> ParamSpace {
        ParamSpace::new()
            .with_integer("iterations", 1, self.max_iterations, false)
            .with_integer("population_size", 4, self.max_population_size, false)
            .with_real("mutation_strength", 0.0, 1.0, false)
            .with_categorical("dimension", self.dimensions_allowed.clone())
    }
}
