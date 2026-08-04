//! Migration selection builders (chooses which individuals are transferred).

use irace_rs::param_space::ParamSpace;
use mahf::{components::selection, params::Params, Component, ExecResult};

use crate::{migrations::ComponentBuilder, problems::RealValuedProblem};

pub mod tournament {
    use super::*;

    #[derive(Clone)]
    pub struct Builder;

    impl Builder {
        pub fn new<P: RealValuedProblem>() -> Box<dyn ComponentBuilder<P>> {
            Box::new(Self)
        }
    }

    impl<P: RealValuedProblem> ComponentBuilder<P> for Builder {
        fn name(&self) -> String {
            "t".to_string()
        }

        fn build(
            &self,
            mut params: Params,
            associated_params: &Params,
        ) -> ExecResult<Box<dyn Component<P>>> {
            let population_size = *associated_params.try_get::<u32>("population_size")? as f64;
            let selection_ratio = params.try_extract::<f64>("selection_ratio")?;
            let num_selected = ((population_size * selection_ratio) as u32).max(1);

            let size_ratio = params.try_extract::<f64>("size_ratio")?;
            let size = ((num_selected as f64 * size_ratio) as u32).max(if num_selected > 1 {
                2
            } else {
                1
            });

            Ok(selection::Tournament::new(num_selected, size))
        }

        fn param_space(&self) -> ParamSpace {
            ParamSpace::new()
                .with_real("selection_ratio", 0.05, 1.0, false)
                .with_real("size_ratio", 0.0, 0.5, false)
        }
    }
}

pub mod random {
    use super::*;

    #[derive(Clone)]
    pub struct Builder;

    impl Builder {
        pub fn new<P: RealValuedProblem>() -> Box<dyn ComponentBuilder<P>> {
            Box::new(Self)
        }
    }

    impl<P: RealValuedProblem> ComponentBuilder<P> for Builder {
        fn name(&self) -> String {
            "r".to_string()
        }

        fn build(
            &self,
            mut params: Params,
            associated_params: &Params,
        ) -> ExecResult<Box<dyn Component<P>>> {
            let population_size = *associated_params.try_get::<u32>("population_size")? as f64;
            let selection_ratio = params.try_extract::<f64>("selection_ratio")?;
            let num_selected = (population_size * selection_ratio) as u32;

            Ok(selection::FullyRandom::new(num_selected))
        }

        fn param_space(&self) -> ParamSpace {
            ParamSpace::new().with_real("selection_ratio", 0.05, 1.0, false)
        }
    }
}
