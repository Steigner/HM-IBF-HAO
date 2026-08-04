use std::marker::PhantomData;

use mahf::{problems::Evaluate, Individual, Problem, State};

pub struct DummyEvaluator<P: Problem>(PhantomData<fn() -> P>);

impl<P: Problem> DummyEvaluator<P> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P: Problem> Default for DummyEvaluator<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Problem> Evaluate for DummyEvaluator<P> {
    type Problem = P;

    fn evaluate(
        &mut self,
        _problem: &Self::Problem,
        _state: &mut State<Self::Problem>,
        _individuals: &mut [Individual<Self::Problem>],
    ) {
        // Noop
    }
}
