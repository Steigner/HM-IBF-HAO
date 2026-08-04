//! Differential Evolution (DE) island builder.

use eyre::bail;
use grahf::problems::algorithm_design::builder::MetaheuristicIslandBuilder;
use irace_rs::param_space::ParamSpace;
use mahf::{params::Params, prelude::*};

use crate::{
    islands::{default_initialization, safe_boundary, safe_de, safe_diversity, ResetIterations},
    problems::{DimensionAwareDomain, RealValuedProblem},
};

/// Island builder for Differential Evolution.
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
    /// Creates a new DE island builder.
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
        "de".to_string()
    }

    fn build_init(&self, params: Params) -> ExecResult<Box<dyn Component<P>>> {
        default_initialization(params)
    }

    fn build_island(&self, mut params: Params) -> ExecResult<Box<dyn Component<P>>> {
        let iterations = params.try_extract::<u32>("iterations")?;

        let y = params.try_extract::<u32>("y")?;

        let selection = match params.try_extract::<String>("selection")?.as_str() {
            "best" => selection::de::DEBest::new(y)?,
            "current2best" => selection::de::DECurrentToBest::new(y)?,
            "rand" => selection::de::DERand::new(y)?,
            _ => bail!("invalid DE selection method"),
        };

        let pc = params.try_extract::<f64>("pc")?;
        // Use safe crossover that uses solution.len() instead of problem.dimension()
        let crossover: Box<dyn Component<P>> =
            match params.try_extract::<String>("crossover")?.as_str() {
                "binomial" => safe_de::SafeDEBinomialCrossover::new(pc),
                "exponential" => safe_de::SafeDEExponentialCrossover::new(pc),
                _ => bail!("invalid DE crossover method"),
            };

        let f = params.try_extract::<f64>("f")?;

        let mutation = mutation::de::DEMutation::new(y, f)?;

        Ok(Configuration::builder()
            .do_(ResetIterations::new())
            .while_(conditions::LessThanN::iterations(iterations), |builder| {
                builder
                    .do_(selection)
                    .do_(mutation)
                    .do_(crossover)
                    .do_(safe_boundary::IslandDimensionSaturation::new())
                    .evaluate()
                    .update_best_individual()
                    .do_(replacement::KeepBetterAtIndex::new())
                    // Use safe diversity that uses solution.len() instead of problem.dimension()
                    .do_(safe_diversity::SafePairwiseDistanceDiversity::new())
                    .do_(logging::Logger::new())
            })
            .build_component())
    }

    fn param_space(&self) -> ParamSpace {
        ParamSpace::new()
            .with_integer("iterations", 1, self.max_iterations, false)
            .with_integer("population_size", 5, self.max_population_size, false)
            .with_categorical_names("selection", ["best", "current2best", "rand"])
            .with_real("pc", 0.0, 1.0, false)
            .with_categorical_names("crossover", ["binomial", "exponential"])
            .with_categorical("y", [1u32, 2u32])
            .with_real("f", 0.0, 2.0, false)
            .with_categorical("dimension", self.dimensions_allowed.clone())
    }
}
