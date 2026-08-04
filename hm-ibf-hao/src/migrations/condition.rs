//! Migration condition builders (decides when to trigger migration).

pub mod random {
    use irace_rs::param_space::ParamSpace;
    use mahf::{conditions::RandomChance, params::Params, Condition, ExecResult};

    use crate::{migrations::ConditionBuilder, problems::RealValuedProblem};

    #[derive(Clone)]
    pub struct Builder;

    impl Builder {
        pub fn new<P: RealValuedProblem>() -> Box<dyn ConditionBuilder<P>> {
            Box::new(Self)
        }
    }

    impl<P: RealValuedProblem> ConditionBuilder<P> for Builder {
        fn name(&self) -> String {
            "r".to_string()
        }

        fn build(
            &self,
            mut params: Params,
            _source_params: &Params,
            _target_params: &Params,
        ) -> ExecResult<Box<dyn Condition<P>>> {
            let p = params.try_extract::<f64>("p")?;
            Ok(RandomChance::new(p))
        }

        fn param_space(&self) -> ParamSpace {
            ParamSpace::new().with_real("p", 0.0, 1.0, false)
        }
    }
}
