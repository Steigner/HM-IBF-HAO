//! Migration replacement builders (decides how incoming individuals replace existing ones).

use irace_rs::param_space::ParamSpace;
use mahf::{components::replacement, params::Params, Component, ExecResult};

use crate::{migrations::ComponentBuilder, problems::RealValuedProblem};

pub mod mu_plus_lambda {
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
            "+".to_string()
        }

        fn build(
            &self,
            _params: Params,
            associated_params: &Params,
        ) -> ExecResult<Box<dyn Component<P>>> {
            let population_size = associated_params.try_get::<u32>("population_size")?;
            Ok(replacement::MuPlusLambda::new(*population_size))
        }

        fn param_space(&self) -> ParamSpace {
            ParamSpace::new()
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
            _params: Params,
            associated_params: &Params,
        ) -> ExecResult<Box<dyn Component<P>>> {
            let population_size = associated_params.try_get::<u32>("population_size")?;
            Ok(replacement::RandomReplacement::new(*population_size))
        }

        fn param_space(&self) -> ParamSpace {
            ParamSpace::new()
        }
    }
}
