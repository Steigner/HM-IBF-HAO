//! Simulated Annealing (SA) island builder.

use grahf::problems::algorithm_design::builder::MetaheuristicIslandBuilder;
use irace_rs::param_space::ParamSpace;
use mahf::{params::Params, prelude::*};

use crate::{
    islands::{default_initialization, make_mutation, safe_boundary, ResetIterations},
    problems::{DimensionAwareDomain, RealValuedProblem},
};

/// Island builder for Simulated Annealing with geometric cooling.
#[derive(Clone)]
pub struct Builder {
    /// Allowed island working dimensions in offsets along the backbone.
    pub dimensions_allowed: Vec<u32>,
    /// Maximum number of iterations IRACE can assign to this island.
    pub max_iterations: u32,
}

impl Builder {
    /// Creates a new SA island builder.
    pub fn new<P: RealValuedProblem + DimensionAwareDomain>(
        dimensions_allowed: &[u32],
        max_iterations: u32,
    ) -> Box<dyn MetaheuristicIslandBuilder<P>> {
        Box::new(Self {
            dimensions_allowed: dimensions_allowed.to_vec(),
            max_iterations,
        })
    }
}

impl<P: RealValuedProblem + DimensionAwareDomain> MetaheuristicIslandBuilder<P> for Builder {
    fn id(&self) -> String {
        "sa".to_string()
    }

    fn build_init(&self, params: Params) -> ExecResult<Box<dyn Component<P>>> {
        default_initialization(params)
    }

    fn build_island(&self, mut params: Params) -> ExecResult<Box<dyn Component<P>>> {
        let iterations = params.try_extract::<u32>("iterations")?;

        let generation = make_mutation(
            &params.try_extract::<String>("mutation")?,
            params.try_extract::<f64>("mutation_strength")?,
        )?;

        let alpha = params.try_extract::<f64>("alpha")?;
        let cooling_schedule = mapping::sa::GeometricCooling::new(
            alpha,
            ValueOf::<replacement::sa::Temperature>::new(),
        )?;

        let t_0 = params.try_extract::<f64>("t_0")?;

        Ok(Configuration::builder()
            .do_(ResetIterations::new())
            .while_(conditions::LessThanN::iterations(iterations), |builder| {
                builder
                    .do_(selection::All::new())
                    .do_(generation)
                    .do_(safe_boundary::IslandDimensionSaturation::new())
                    .evaluate()
                    .update_best_individual()
                    .do_(cooling_schedule)
                    .do_(replacement::sa::ExponentialAnnealingAcceptance::new(t_0))
                    .do_(logging::Logger::new())
            })
            .build_component())
    }

    fn param_space(&self) -> ParamSpace {
        ParamSpace::new()
            .with_integer("iterations", 1, self.max_iterations, false)
            .with_categorical("population_size", [1u32])
            .with_real("mutation_strength", 1e-3, 0.5, true)
            .with_categorical_names("mutation", ["normal", "uniform"])
            .with_real("t_0", 1e-3, 1000.0, true)
            .with_real("alpha", 0.9, 1.0 - 1e-4, false)
            .with_categorical("dimension", self.dimensions_allowed.clone())
    }
}
