use dyn_clone::DynClone;
use irace_rs::param_space::ParamSpace;
use mahf::{
    logging, params::Params, Component, Condition, Configuration, ExecResult, Problem,
    SingleObjectiveProblem,
};
use petgraph::prelude::EdgeRef;

use crate::{components::island, graph::DiGraph, problems::algorithm_design::islands::Migration};

pub trait MetaheuristicIslandBuilder<P: Problem>: DynClone + Send + Sync + 'static {
    fn id(&self) -> String;

    fn build_init(&self, params: Params) -> ExecResult<Box<dyn Component<P>>>;

    fn build_island(&self, params: Params) -> ExecResult<Box<dyn Component<P>>>;

    fn param_space(&self) -> ParamSpace;
}

dyn_clone::clone_trait_object!(<P: Problem> MetaheuristicIslandBuilder<P>);

pub trait MigrationBuilder<P: Problem>: DynClone + Send + Sync + 'static {
    fn id(&self) -> String;

    fn build_migration(
        &self,
        params: Params,
        source_params: &Params,
        target_params: &Params,
    ) -> ExecResult<Migration<P>>;

    fn param_space(&self) -> ParamSpace;
}

dyn_clone::clone_trait_object!(<P: Problem> MigrationBuilder<P>);

pub type BuilderGraph<P> =
    DiGraph<Box<dyn MetaheuristicIslandBuilder<P>>, Box<dyn MigrationBuilder<P>>>;

pub trait MetaheuristicBuilder<P: SingleObjectiveProblem>:
    Fn(Params) -> ExecResult<Configuration<P>> + Clone + Send + 'static
{
}

impl<P: SingleObjectiveProblem, F> MetaheuristicBuilder<P> for F where
    F: Fn(Params) -> ExecResult<Configuration<P>> + Clone + Send + 'static
{
}

impl<P: SingleObjectiveProblem> BuilderGraph<P> {
    pub fn param_space(&self) -> ParamSpace {
        let mut island_param_space = ParamSpace::new();
        for (i, builder) in self.node_references() {
            island_param_space.add_nested(i.index().to_string(), builder.param_space());
        }

        let mut migration_param_space = ParamSpace::new();
        for edge in self.edge_references() {
            migration_param_space
                .add_nested(edge.id().index().to_string(), edge.weight().param_space());
        }

        let mut param_space = ParamSpace::new()
            .with_nested("island", island_param_space)
            .with_nested("migration", migration_param_space);

        param_space.flatten();

        param_space
    }

    pub fn into_builder(self, condition: Box<dyn Condition<P>>) -> impl MetaheuristicBuilder<P>
    where
        P::Encoding: AsRef<[f64]>,
    {
        // Builder method for `Configuration`s from `Params`.
        move |mut params: Params| -> ExecResult<Configuration<P>> {
            params.nest();

            // Split builder graph into initializations and island graphs.
            let mut initializations = Vec::new();

            let island_graph = self.try_map(
                |node, builder| {
                    let island_params = params
                        .try_get::<Params>("island")?
                        .try_get::<Params>(&node.index().to_string())
                        .cloned()
                        .unwrap_or_default();

                    initializations.push(builder.build_init(island_params.clone())?);
                    builder.build_island(island_params)
                },
                |edge, builder| {
                    let migration_params = params
                        .try_get::<Params>("migration")?
                        .try_get::<Params>(&edge.index().to_string())
                        .cloned()?;
                    let (source, target) = self.edge_endpoints(edge).unwrap();
                    let source_params = params
                        .try_get::<Params>("island")?
                        .try_get::<Params>(&source.index().to_string())?;
                    let target_params = params
                        .try_get::<Params>("island")?
                        .try_get::<Params>(&target.index().to_string())?;

                    builder.build_migration(migration_params, source_params, target_params)
                },
            )?;

            // Construct the algorithm.
            let algorithm = Configuration::builder()
                .do_(island::IslandStatesInit::new(initializations))
                .while_(condition.clone(), |builder| {
                    builder
                        .do_(island::IslandGraphExecutor::new(island_graph))
                        .do_(island::UpdateBestIslandIndividual::new())
                        .do_(logging::Logger::new())
                })
                .build();

            Ok(algorithm)
        }
    }
}
