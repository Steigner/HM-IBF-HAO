//! Integration tests covering the horizontal alignment benchmark end to end.
//!
//! These drive the crate through its public API: load the shipped instances, score
//! alignments, migrate solutions between island dimensions and export a run.

use std::{fs, path::PathBuf};

use grahf::components::transform::{SolutionTransformer, TransformRequest};
use grahf_hao::{
    alignment::{
        output::OUTPUT_TRANSFORM_NOT_APPLIED, write_run_results, HorizontalAlignment, RunMetadata,
    },
    config::{TrainingParams, DEFAULT_TRAINING_PARAMS},
    islands::transforms::{OffsetResampleTransformer, TransformMethod},
    problems::DimensionAwareDomain,
};
use mahf::{
    problems::{KnownOptimumProblem, LimitedVectorProblem, VectorProblem},
    Problem, Random,
};

/// Returns the directory holding the instances shipped with the crate.
fn instances_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("instances")
}

/// Returns the shipped `params_training.conf`'s allowed island dimensions.
fn dimensions_allowed() -> Vec<u32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_TRAINING_PARAMS);
    TrainingParams::load(&path)
        .expect("the shipped params_training.conf must load")
        .dimensions_allowed
}

/// Loads every shipped instance.
fn instances() -> Vec<HorizontalAlignment> {
    HorizontalAlignment::load_instances(instances_dir(), &dimensions_allowed())
        .expect("the shipped instances must load")
}

/// Builds a deterministic offset vector of the given dimension.
fn offsets(dimension: usize) -> Vec<f64> {
    (0..dimension).map(|i| ((i as f64) * 0.37).sin()).collect()
}

#[test]
fn every_shipped_instance_loads_and_is_self_consistent() {
    let instances = instances();

    assert!(!instances.is_empty(), "the crate must ship instances");
    for instance in &instances {
        let points = instance.backbone_points.len();
        assert!(
            points >= 2,
            "{}: backbone of {points} points",
            instance.name
        );
        assert_eq!(instance.backbone_normals.len(), points, "{}", instance.name);
        assert_eq!(
            instance.backbone_offset_bounds.len(),
            points,
            "{}",
            instance.name
        );
        assert_eq!(
            instance.backbone_cumulative_distances.len(),
            points,
            "{}",
            instance.name
        );
        assert!(
            instance.backbone_total_length > 0.0,
            "{}: empty backbone",
            instance.name
        );
        assert!(
            instance
                .backbone_points
                .iter()
                .all(|point| point.iter().all(|value| value.is_finite())),
            "{} has a non-finite backbone point",
            instance.name
        );
        assert_eq!(instance.name(), instance.name);
    }
}

#[test]
fn the_summary_file_is_not_loaded_as_an_instance() {
    assert!(instances_dir().join("summary.json").exists());
    assert!(instances().iter().all(|i| i.name() != "summary"));
}

#[test]
fn the_instances_are_loaded_in_a_deterministic_order() {
    let first: Vec<String> = instances().iter().map(|i| i.name.clone()).collect();
    let second: Vec<String> = instances().iter().map(|i| i.name.clone()).collect();

    assert_eq!(first, second);
    let mut sorted = first.clone();
    sorted.sort();
    assert_eq!(first, sorted, "instances must load in name order");
}

#[test]
fn the_objective_is_finite_and_non_negative_for_every_allowed_dimension() {
    for instance in instances() {
        for &dimension in &dimensions_allowed() {
            let value = instance.evaluate_solution(&offsets(dimension as usize));

            assert!(
                value.is_finite() && value >= 0.0,
                "{} at D={dimension} produced {value}",
                instance.name
            );
            assert!(
                value >= instance.known_optimum().value(),
                "{} at D={dimension} beat the known optimum",
                instance.name
            );
        }
    }
}

#[test]
fn a_zero_offset_alignment_follows_the_backbone() {
    // With every offset at zero the corridor is the backbone itself, so its weighted length
    // is at least the backbone's arc length - the terrain can only make it more expensive.
    for instance in instances() {
        let value = instance.evaluate_solution(&vec![0.0; dimensions_allowed()[0] as usize]);

        assert!(
            value >= instance.backbone_total_length * 0.5,
            "{}: {value} is implausibly short for a backbone of {}",
            instance.name,
            instance.backbone_total_length
        );
    }
}

#[test]
fn the_control_polygon_brackets_the_backbone_and_stays_on_the_terrain() {
    for instance in instances() {
        let (rows, columns) = instance.heightmap.dim();

        for &dimension in &dimensions_allowed() {
            let polygon = instance.control_polygon(&offsets(dimension as usize));

            assert_eq!(polygon.len(), dimension as usize + 2, "{}", instance.name);
            assert_eq!(polygon[0], instance.backbone_points[0]);
            assert_eq!(
                *polygon.last().unwrap(),
                *instance.backbone_points.last().unwrap()
            );
            for point in &polygon[1..polygon.len() - 1] {
                assert!(
                    (0.0..rows as f64).contains(&point[0])
                        && (0.0..columns as f64).contains(&point[1]),
                    "{}: {point:?} left the heightmap",
                    instance.name
                );
            }
        }
    }
}

#[test]
fn the_declared_domain_covers_the_largest_island_dimension() {
    let instance = &instances()[0];

    assert_eq!(
        instance.dimension(),
        *dimensions_allowed().last().unwrap() as usize
    );
    assert_eq!(instance.domain().len(), instance.dimension());
    for &dimension in &dimensions_allowed() {
        assert_eq!(
            instance.domain_for_dimension(dimension as usize).len(),
            dimension as usize
        );
    }
}

#[test]
fn migration_between_any_two_allowed_dimensions_yields_a_valid_solution() {
    let instance = &instances()[0];
    let transformer = OffsetResampleTransformer::new();
    let mut rng = Random::new(0);

    let dimensions_allowed = dimensions_allowed();
    for &source in &dimensions_allowed {
        for &target in &dimensions_allowed {
            for method in TransformMethod::all_names() {
                let input = offsets(source as usize);
                let request = TransformRequest::new(source, target, method);

                let output = transformer.transform(instance, &input, request, &mut rng);

                assert_eq!(
                    output.len(),
                    target as usize,
                    "{method}: {source} -> {target}"
                );
                assert!(
                    instance.evaluate_solution(&output).is_finite(),
                    "{method}: {source} -> {target} is not evaluable"
                );
            }
        }
    }
}

#[test]
fn migrated_solutions_stay_inside_the_target_dimension_bounds() {
    let instance = &instances()[0];
    let transformer = OffsetResampleTransformer::new();
    let mut rng = Random::new(1);
    let dimensions_allowed = dimensions_allowed();
    let (source, target) = (dimensions_allowed[0], *dimensions_allowed.last().unwrap());
    let bounds = instance.domain_for_dimension(target as usize);

    for method in TransformMethod::all_names() {
        let input = vec![1e6; source as usize];

        let output = transformer.transform(
            instance,
            &input,
            TransformRequest::new(source, target, method),
            &mut rng,
        );

        for (offset, range) in output.iter().zip(&bounds) {
            assert!(
                (range.start..=range.end).contains(offset),
                "{method} produced {offset} outside [{}, {}]",
                range.start,
                range.end
            );
        }
    }
}

#[test]
fn exporting_a_run_produces_a_readable_payload() {
    let dir = tempfile::tempdir().unwrap();
    let instance = &instances()[0];
    let solution = offsets(dimensions_allowed()[0] as usize);
    let best_value = instance.evaluate_solution(&solution);

    let metadata = RunMetadata {
        problem: instance.name(),
        natural_dimension: instance.natural_dimension,
        backbone_length: instance.backbone_total_length,
        solution_dim: solution.len(),
        output_transform_method: OUTPUT_TRANSFORM_NOT_APPLIED,
        solution: &solution,
    };

    let run_dir =
        write_run_results(dir.path(), &metadata, "GRAHF", 42, 100_000, best_value).unwrap();

    let payload: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(run_dir.join("results.json")).unwrap()).unwrap();

    // The exported `x` must be the vector that produced the exported `best_value`.
    //
    // Bit equality is not asserted: the JSON float parser can be one ULP off when reading
    // back a value its own writer emitted, so the round trip is checked against a tolerance
    // far below anything that affects the objective.
    let exported: Vec<f64> = payload["x"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();

    assert_eq!(exported.len(), solution.len());
    for (before, after) in solution.iter().zip(&exported) {
        assert!(
            (before - after).abs() <= f64::EPSILON * before.abs().max(1.0),
            "{before} round-tripped to {after}"
        );
    }

    let reported = payload["best_value"].as_f64().unwrap();
    assert!(
        (instance.evaluate_solution(&exported) - reported).abs() < 1e-9,
        "recomputing f(x) from the export must reproduce the reported value"
    );
    assert_eq!(
        payload["solution_dim"].as_u64().unwrap() as usize,
        solution.len()
    );
    assert_eq!(
        payload["natural_dimension"].as_u64().unwrap() as usize,
        instance.natural_dimension
    );
}
