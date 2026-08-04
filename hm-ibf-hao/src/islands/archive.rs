//! Archive island builder — a passive island that only tracks diversity, no search.

use grahf::problems::algorithm_design::builder::MetaheuristicIslandBuilder;
use irace_rs::param_space::ParamSpace;
use mahf::{params::Params, prelude::*};

use crate::{
    islands::{default_initialization, safe_diversity},
    problems::{DimensionAwareDomain, RealValuedProblem},
};

/// Island builder for a passive archive that measures population diversity without evolving.
#[derive(Clone)]
pub struct Builder {
    /// Allowed island working dimensions in offsets along the backbone.
    pub dimensions_allowed: Vec<u32>,
    /// Maximum population size IRACE can assign to this island.
    pub max_population_size: u32,
}

impl Builder {
    /// Creates a new archive island builder.
    pub fn new<P: RealValuedProblem + DimensionAwareDomain>(
        dimensions_allowed: &[u32],
        max_population_size: u32,
    ) -> Box<dyn MetaheuristicIslandBuilder<P>> {
        Box::new(Self {
            dimensions_allowed: dimensions_allowed.to_vec(),
            max_population_size,
        })
    }
}

impl<P: RealValuedProblem + DimensionAwareDomain> MetaheuristicIslandBuilder<P> for Builder {
    fn id(&self) -> String {
        "ar".to_string()
    }

    fn build_init(&self, params: Params) -> ExecResult<Box<dyn Component<P>>> {
        default_initialization(params)
    }

    fn build_island(&self, _params: Params) -> ExecResult<Box<dyn Component<P>>> {
        Ok(Configuration::builder()
            .do_(safe_diversity::SafePairwiseDistanceDiversity::new())
            .do_(logging::Logger::new())
            .build_component())
    }

    fn param_space(&self) -> ParamSpace {
        ParamSpace::new()
            .with_integer("population_size", 5, self.max_population_size, false)
            .with_categorical("dimension", self.dimensions_allowed.clone())
    }
}
