//! Island builder collection for the horizontal alignment problem.
//!
//! Each builder produces a MAHF island (DE, ES, LS, SA, RS, or Archive) wired to work at
//! any IRACE-tuned dimension via `RandomSpreadWithDimension` initialization.
//!
//! PSO is deliberately absent. It carries per-particle auxiliary state — a velocity and a
//! personal best — that has no component-wise correspondent once a migration changes an
//! island's working dimension, so it belongs to the homogeneous special case rather than to
//! the dimension-agnostic island set.

#![allow(clippy::new_ret_no_self)]

use better_any::{Tid, TidAble};
use eyre::bail;
use grahf::problems::algorithm_design::builder::MetaheuristicIslandBuilder;
use mahf::population::IntoIndividuals;
use mahf::{params::Params, prelude::*};
use serde::{Deserialize, Serialize};

use crate::problems::{DimensionAwareDomain, RealValuedProblem};

pub mod archive;
pub mod de;
pub mod es;
pub mod ls;
pub mod rs;
pub mod sa;
pub mod safe_boundary;
pub mod safe_de;
pub mod safe_diversity;
pub mod transforms;

/// Node weight of the differential evolution island.
pub const ISLAND_DE: u32 = 0;
/// Node weight of the evolution strategy island.
pub const ISLAND_ES: u32 = 1;
/// Node weight of the local search island.
pub const ISLAND_LS: u32 = 2;
/// Node weight of the simulated annealing island.
pub const ISLAND_SA: u32 = 3;
/// Node weight of the random search island.
pub const ISLAND_RS: u32 = 4;
/// Node weight of the passive archive island.
pub const ISLAND_ARCHIVE: u32 = 5;

pub use transforms::{transform_with_optional_params, TransformMethod};

/// Returns all island builders GRAHF may place on a node.
///
/// The order is the node weight encoding and is mirrored by the `ISLAND_*` constants;
/// changing it invalidates every stored `elitist_*.json`.
///
/// # Arguments
///
/// * `dimensions_allowed` - Allowed island working dimensions; see
///   [`crate::config::TrainingParams::dimensions_allowed`].
/// * `max_iterations` - Upper bound IRACE may assign to an island's iteration count.
/// * `max_population_size` - Upper bound IRACE may assign to an island's population size.
///
/// # Returns
///
/// The builders, indexed by node weight.
pub fn island_builders<P: RealValuedProblem + DimensionAwareDomain>(
    dimensions_allowed: &[u32],
    max_iterations: u32,
    max_population_size: u32,
) -> Vec<Box<dyn MetaheuristicIslandBuilder<P>>> {
    vec![
        de::Builder::new(dimensions_allowed, max_iterations, max_population_size),
        es::Builder::new(dimensions_allowed, max_iterations, max_population_size),
        ls::Builder::new(dimensions_allowed, max_iterations, max_population_size),
        sa::Builder::new(dimensions_allowed, max_iterations),
        rs::Builder::new(dimensions_allowed, max_iterations, max_population_size),
        archive::Builder::new(dimensions_allowed, max_population_size),
    ]
}

/// Builds the initialization shared by the islands that need no extra state.
///
/// # Arguments
///
/// * `params` - IRACE parameters; `population_size` is required, `dimension` optional.
///
/// # Returns
///
/// A component that spreads a population across the domain and evaluates it.
///
/// # Errors
///
/// Returns an error if `population_size` is missing from `params`.
pub fn default_initialization<P: RealValuedProblem + DimensionAwareDomain>(
    mut params: Params,
) -> ExecResult<Box<dyn Component<P>>> {
    let population_size = params.try_extract::<u32>("population_size")?;
    let dimension = params.try_extract::<u32>("dimension").ok();

    Ok(Configuration::builder()
        .do_(RandomSpreadWithDimension::new(population_size, dimension))
        .evaluate()
        .update_best_individual()
        .build_component())
}

/// Builds a mutation component by name.
///
/// # Arguments
///
/// * `name` - The mutation method, either `"normal"` or `"uniform"`.
/// * `strength` - The mutation strength.
///
/// # Returns
///
/// The mutation component.
///
/// # Errors
///
/// Returns an error if `name` is not a known mutation method.
pub fn make_mutation<P: RealValuedProblem>(
    name: &str,
    strength: f64,
) -> ExecResult<Box<dyn Component<P>>> {
    let mutation = match name {
        "normal" => mutation::NormalMutation::new(strength, 1.0),
        "uniform" => mutation::UniformMutation::new(strength, 1.0),
        _ => bail!("invalid mutation method: {}", name),
    };
    Ok(mutation)
}

/// Resets the number of iterations.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ResetIterations;

impl ResetIterations {
    /// Creates the component.
    ///
    /// # Returns
    ///
    /// The component value.
    pub fn from_params() -> Self {
        Self
    }

    /// Creates the component, boxed for use in a configuration.
    ///
    /// # Returns
    ///
    /// The boxed component.
    pub fn new<P: Problem>() -> Box<dyn Component<P>> {
        Box::new(Self::from_params())
    }
}

impl<P: Problem> Component<P> for ResetIterations {
    fn execute(&self, _problem: &P, state: &mut State<P>) -> ExecResult<()> {
        state.set_value::<common::Iterations>(0);
        Ok(())
    }
}

/// Initializes a population spread uniformly across the domain at a specific dimension.
///
/// When `dimension` is `None`, falls back to `problem.dimension()`. When set, each solution
/// has exactly that many elements regardless of the problem's declared dimension.
#[derive(Clone, Serialize, Deserialize)]
pub struct RandomSpreadWithDimension {
    /// Number of individuals in the initial population.
    pub population_size: u32,
    /// Working dimension for this island; `None` means use `problem.dimension()`.
    pub dimension: Option<u32>,
}

impl RandomSpreadWithDimension {
    /// Creates the component.
    ///
    /// # Arguments
    ///
    /// * `population_size` - Number of individuals to sample.
    /// * `dimension` - Working dimension, or `None` to use the problem's own.
    ///
    /// # Returns
    ///
    /// The component value.
    pub fn from_params(population_size: u32, dimension: Option<u32>) -> Self {
        Self {
            population_size,
            dimension,
        }
    }

    /// Creates the component, boxed for use in a configuration.
    ///
    /// # Arguments
    ///
    /// * `population_size` - Number of individuals to sample.
    /// * `dimension` - Working dimension, or `None` to use the problem's own.
    ///
    /// # Returns
    ///
    /// The boxed component.
    pub fn new<P: RealValuedProblem + DimensionAwareDomain>(
        population_size: u32,
        dimension: Option<u32>,
    ) -> Box<dyn Component<P>> {
        Box::new(Self::from_params(population_size, dimension))
    }
}

impl<P: RealValuedProblem + DimensionAwareDomain> Component<P> for RandomSpreadWithDimension {
    fn init(&self, problem: &P, state: &mut State<P>) -> ExecResult<()> {
        let dim = self
            .dimension
            .map(|d| d as usize)
            .unwrap_or_else(|| problem.dimension());

        state.insert(IslandDimension(dim));

        // Call `domain_for_dimension(dim)` so each island gets bounds that correspond to
        // its actual control-point positions on the backbone.  For `HorizontalAlignment`
        // this resamples the backbone at `s_i = i * L / (D+1)` for the exact `dim` used,
        // rather than reusing the max-dimension bounds (which sit at different positions).
        let domain = problem.domain_for_dimension(dim);

        let mut rng = state.random_mut();

        let population: Vec<_> = (0..self.population_size)
            .map(|_| {
                let solution: Vec<f64> = domain
                    .iter()
                    .map(|range| {
                        use mahf::rand::Rng;
                        rng.gen_range(range.clone())
                    })
                    .collect();
                solution
            })
            .into_individuals();

        state.populations_mut().push(population);
        Ok(())
    }

    fn execute(&self, _problem: &P, _state: &mut State<P>) -> ExecResult<()> {
        Ok(())
    }
}

/// The actual working dimension of this island, stored in island state.
///
/// May differ from `problem.dimension()` when IRACE tunes per-island dimensions.
/// Safe components read this instead of calling `problem.dimension()`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Tid)]
pub struct IslandDimension(pub usize);

impl<'a> mahf::CustomState<'a> for IslandDimension {}

impl std::ops::Deref for IslandDimension {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::HorizontalAlignment;

    /// Allowed island dimensions used by the tests in this module.
    ///
    /// The real invariants (non-empty, strictly increasing, positive) are enforced at load
    /// time by `config::validate_dimensions_allowed` and checked against the shipped
    /// `params_training.conf` there; these tests only need *some* valid list.
    const TEST_DIMENSIONS: [u32; 3] = [53, 60, 68];

    #[test]
    fn the_island_constants_match_the_builder_order() {
        let builders = island_builders::<HorizontalAlignment>(&TEST_DIMENSIONS, 10, 10);
        let ids: Vec<String> = builders.iter().map(|builder| builder.id()).collect();

        assert_eq!(ids[ISLAND_DE as usize], "de");
        assert_eq!(ids[ISLAND_ES as usize], "es");
        assert_eq!(ids[ISLAND_LS as usize], "ls");
        assert_eq!(ids[ISLAND_SA as usize], "sa");
        assert_eq!(ids[ISLAND_RS as usize], "rs");
        assert_eq!(ids[ISLAND_ARCHIVE as usize], "ar");
        assert_eq!(ids.len(), 6, "every island must have a constant");
    }

    #[test]
    fn the_island_ids_are_unique() {
        let builders = island_builders::<HorizontalAlignment>(&TEST_DIMENSIONS, 10, 10);
        let mut ids: Vec<String> = builders.iter().map(|builder| builder.id()).collect();
        ids.sort();
        let count = ids.len();
        ids.dedup();

        assert_eq!(ids.len(), count, "duplicate island ids: {ids:?}");
    }

    #[test]
    fn the_dimension_agnostic_island_set_excludes_pso() {
        let builders = island_builders::<HorizontalAlignment>(&TEST_DIMENSIONS, 10, 10);
        let ids: Vec<String> = builders.iter().map(|builder| builder.id()).collect();

        assert!(
            !ids.iter().any(|id| id == "pso"),
            "PSO's auxiliary per-particle state has no meaning across a dimension change: {ids:?}"
        );
    }

    #[test]
    fn an_unknown_mutation_method_is_rejected() {
        // `Box<dyn Component>` is not `Debug`, so `unwrap_err` cannot be used here.
        let error = match make_mutation::<HorizontalAlignment>("quantum", 0.1) {
            Ok(_) => panic!("expected the unknown method to be rejected"),
            Err(error) => error,
        };

        assert!(
            error.to_string().contains("invalid mutation method"),
            "{error}"
        );
    }

    #[test]
    fn the_known_mutation_methods_are_built() {
        assert!(make_mutation::<HorizontalAlignment>("normal", 0.1).is_ok());
        assert!(make_mutation::<HorizontalAlignment>("uniform", 0.1).is_ok());
    }

    #[test]
    fn the_island_dimension_derefs_to_its_value() {
        let dimension = IslandDimension(13);

        assert_eq!(*dimension, 13);
    }
}
