//! Evolution Strategy (ES) island builder.

use eyre::bail;
use grahf::problems::algorithm_design::builder::MetaheuristicIslandBuilder;
use irace_rs::param_space::ParamSpace;
use mahf::{params::Params, prelude::*};

use crate::{
    islands::{
        default_initialization, make_mutation, safe_boundary, safe_diversity, ResetIterations,
    },
    problems::{DimensionAwareDomain, RealValuedProblem},
};

/// Island builder for Evolution Strategy with tunable selection and replacement.
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
    /// Creates a new ES island builder.
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
        "es".to_string()
    }

    fn build_init(&self, params: Params) -> ExecResult<Box<dyn Component<P>>> {
        default_initialization(params)
    }

    fn build_island(&self, mut params: Params) -> ExecResult<Box<dyn Component<P>>> {
        let iterations = params.try_extract::<u32>("iterations")?;
        let population_size = params.try_extract::<u32>("population_size")?;

        let selection = match params.try_extract::<String>("selection")?.as_str() {
            "fully_random" => selection::FullyRandom::new(population_size),
            "roulette_wheel" => selection::RouletteWheel::new(population_size, 0.01),
            "tournament" => selection::Tournament::new(population_size, 2),
            "linear_rank" => selection::LinearRank::new(population_size),
            _ => bail!("invalid ES selection method"),
        };

        let mutation = make_mutation(
            &params.try_extract::<String>("mutation")?,
            params.try_extract::<f64>("mutation_strength")?,
        )?;

        let replacement = match params.try_extract::<String>("replacement")?.as_str() {
            "mu+lambda" => replacement::MuPlusLambda::new(population_size),
            "generational" => replacement::Generational::new(population_size),
            "random" => replacement::RandomReplacement::new(population_size),
            _ => bail!("invalid ES replacement method"),
        };

        Ok(Configuration::builder()
            .do_(ResetIterations::new())
            .while_(conditions::LessThanN::iterations(iterations), |builder| {
                builder
                    .do_(selection)
                    .do_(mutation)
                    .do_(safe_boundary::IslandDimensionSaturation::new())
                    .evaluate()
                    .update_best_individual()
                    .do_(replacement)
                    .do_(safe_diversity::SafePairwiseDistanceDiversity::new())
                    .do_(logging::Logger::new())
            })
            .build_component())
    }

    fn param_space(&self) -> ParamSpace {
        ParamSpace::new()
            .with_integer("iterations", 1, self.max_iterations, false)
            .with_integer("population_size", 4, self.max_population_size, false)
            .with_categorical_names(
                "selection",
                [
                    "fully_random",
                    "roulette_wheel",
                    "tournament",
                    "linear_rank",
                ],
            )
            .with_real("mutation_strength", 1e-3, 0.5, true)
            .with_categorical_names("mutation", ["normal", "uniform"])
            .with_categorical_names("replacement", ["mu+lambda", "generational", "random"])
            .with_categorical("dimension", self.dimensions_allowed.clone())
    }
}
