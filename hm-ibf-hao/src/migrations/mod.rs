//! Migration builder collection: combines condition, selection, and replacement builders
//! into the full set of migration policies available to GRAHF.

#![allow(clippy::new_ret_no_self)]

use grahf::problems::algorithm_design::{builder::MigrationBuilder, islands::Migration};
use irace_rs::param_space::ParamSpace;
use itertools::iproduct;
use mahf::{params::Params, Component, Condition, ExecResult, Problem};

use crate::islands::transforms::TransformMethod;
use crate::problems::RealValuedProblem;

pub mod condition;
pub mod replacement;
pub mod selection;

/// Returns the Cartesian product of all condition × selection × replacement builders.
pub fn migration_builders<P: RealValuedProblem>() -> Vec<Box<dyn MigrationBuilder<P>>> {
    let condition_builders = vec![condition::random::Builder::new()];

    let selection_builders = vec![
        selection::tournament::Builder::new(),
        selection::random::Builder::new(),
    ];

    let replacement_builders = vec![
        replacement::mu_plus_lambda::Builder::new(),
        replacement::random::Builder::new(),
    ];

    iproduct!(condition_builders, selection_builders, replacement_builders)
        .map(|(cb, sb, rb)| {
            Box::new(MigrationBuilderBuilder {
                condition_builder: cb,
                selection_builder: sb,
                replacement_builder: rb,
            }) as Box<dyn MigrationBuilder<_>>
        })
        .collect()
}

/// Builds a selection or replacement component from IRACE parameters.
pub trait ComponentBuilder<P: Problem>: dyn_clone::DynClone + Send + Sync + 'static {
    /// Short name used as IRACE parameter namespace.
    fn name(&self) -> String;

    /// Builds the component from its own parameters plus the associated island parameters.
    fn build(
        &self,
        params: Params,
        associated_params: &Params,
    ) -> ExecResult<Box<dyn Component<P>>>;

    /// Declares the parameter space IRACE will tune.
    fn param_space(&self) -> ParamSpace;
}

dyn_clone::clone_trait_object!(<P: Problem> ComponentBuilder<P>);

/// Builds a migration condition from IRACE parameters plus source/target island parameters.
pub trait ConditionBuilder<P: Problem>: dyn_clone::DynClone + Send + Sync + 'static {
    /// Short name used as IRACE parameter namespace.
    fn name(&self) -> String;

    /// Builds the condition, with access to both source and target island parameters.
    fn build(
        &self,
        params: Params,
        source_params: &Params,
        target_params: &Params,
    ) -> ExecResult<Box<dyn Condition<P>>>;

    /// Declares the parameter space IRACE will tune.
    fn param_space(&self) -> ParamSpace;
}

dyn_clone::clone_trait_object!(<P: Problem> ConditionBuilder<P>);

pub struct MigrationBuilderBuilder<P> {
    pub condition_builder: Box<dyn ConditionBuilder<P>>,
    pub selection_builder: Box<dyn ComponentBuilder<P>>,
    pub replacement_builder: Box<dyn ComponentBuilder<P>>,
}

impl<P: Problem> Clone for MigrationBuilderBuilder<P> {
    fn clone(&self) -> Self {
        Self {
            condition_builder: self.condition_builder.clone(),
            selection_builder: self.selection_builder.clone(),
            replacement_builder: self.replacement_builder.clone(),
        }
    }
}

impl<P: Problem> MigrationBuilder<P> for MigrationBuilderBuilder<P> {
    fn id(&self) -> String {
        format!(
            "{}{}{}",
            self.condition_builder.name(),
            self.selection_builder.name(),
            self.replacement_builder.name()
        )
    }

    fn build_migration(
        &self,
        mut params: Params,
        source_params: &Params,
        target_params: &Params,
    ) -> ExecResult<Migration<P>> {
        let condition_params = params
            .try_extract::<Params>("condition")
            .unwrap_or_default();
        let selection_params = params
            .try_extract::<Params>("selection")
            .unwrap_or_default();
        let replacement_params = params
            .try_extract::<Params>("replacement")
            .unwrap_or_default();

        // Extract transform_method for dimension transformation during migration
        let transform_method = params.try_extract::<String>("transform_method").ok();

        // Extract dimensions from source and target params
        let source_dimension = source_params.clone().try_extract::<u32>("dimension").ok();
        let target_dimension = target_params.clone().try_extract::<u32>("dimension").ok();

        Ok(Migration {
            condition: self.condition_builder.build(
                condition_params,
                source_params,
                target_params,
            )?,
            selection: self
                .selection_builder
                .build(selection_params, source_params)?,
            replacement: self
                .replacement_builder
                .build(replacement_params, target_params)?,
            transform_method,
            source_dimension,
            target_dimension,
        })
    }

    fn param_space(&self) -> ParamSpace {
        ParamSpace::new()
            .with_nested("condition", self.condition_builder.param_space())
            .with_nested("selection", self.selection_builder.param_space())
            .with_nested("replacement", self.replacement_builder.param_space())
            .with_categorical_names("transform_method", TransformMethod::all_names())
    }
}
