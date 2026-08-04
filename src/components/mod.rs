//! Graph components.

#![allow(clippy::new_ret_no_self)]

pub mod initialization;
pub mod island;
pub mod mutation;
pub mod normalization;
pub mod recombination;
pub mod transform;

// Re-export key types for convenience
pub use island::{
    IslandGraphExecutor, IslandStates, IslandStatesInit, MigrationTransformer,
    UpdateBestIslandIndividual,
};
