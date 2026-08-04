//! Horizontal alignment optimization experiments using the GRAHF framework.
//!
//! The crate provides the problem definition, island builders, migration builders, the
//! clothoid spline interpolation and the GRAHF genetic algorithm wired together to route a
//! road or railway corridor across a terrain heightmap, plus the training and evaluation
//! stages driven by the `hm-ibf-hao` binary.

pub mod alignment;
pub mod cli;
pub mod clothoid;
pub mod config;
pub mod evaluation;
pub mod heuristic;
pub mod islands;
pub mod migrations;
pub mod problems;
pub mod training;
