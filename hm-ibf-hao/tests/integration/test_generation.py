"""End-to-end generation of an instance set.

Drives :mod:`preprocessing.prepare_instances` the way the runbook does and checks that what
lands on disk is exactly what the Rust benchmark loads.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from preprocessing import prepare_instances

#: Every top-level key the Rust `AlignmentConfig` deserializes.
REQUIRED_KEYS = (
    "name",
    "heightmap_shape",
    "start",
    "goal",
    "path_astar",
    "path_simplified",
    "backbone",
    "epsilon",
    "cutting_plane_factor",
    "tunnel_factor",
    "gradient_factor",
    "curvature_radius",
    "gradient_change_limit",
    "height_limit",
    "tau",
)


@pytest.fixture
def generated(tmp_path, terrain, small_config):
    """Generate both instance sets once and hand back their directories."""
    train_dir = tmp_path / "instances_train"
    eval_dir = tmp_path / "instances_eval"

    dimensions = prepare_instances.generate(terrain, small_config, train_dir, eval_dir)

    return train_dir, eval_dir, dimensions


def load_configs(directory):
    """Read every instance config in a directory, in name order."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*_config.json"))
    ]


def test_both_sets_hold_the_requested_number_of_instances(generated, small_config):
    train_dir, eval_dir, _ = generated

    assert len(load_configs(train_dir)) == small_config.n_train
    assert len(load_configs(eval_dir)) == small_config.n_eval


def test_every_config_is_paired_with_its_heightmap(generated):
    train_dir, eval_dir, _ = generated

    for directory in (train_dir, eval_dir):
        for config_path in directory.glob("*_config.json"):
            name = config_path.name.removesuffix("_config.json")
            assert (directory / f"{name}_heightmap.npy").is_file()


def test_every_config_carries_the_keys_the_loader_reads(generated):
    train_dir, _, _ = generated

    for config in load_configs(train_dir):
        for key in REQUIRED_KEYS:
            assert key in config, key


def test_the_backbone_arrays_agree_in_length(generated):
    train_dir, eval_dir, _ = generated

    for directory in (train_dir, eval_dir):
        for config in load_configs(directory):
            backbone = config["backbone"]
            count = len(backbone["points"])
            assert count >= 2
            assert len(backbone["normals"]) == count
            assert len(backbone["offset_bounds"]) == count
            assert len(backbone["cumulative_distances"]) == count
            assert backbone["total_length"] > 0.0


def test_every_backbone_point_carries_two_coordinates(generated):
    train_dir, _, _ = generated

    for config in load_configs(train_dir):
        backbone = config["backbone"]
        assert all(len(point) == 2 for point in backbone["points"])
        assert all(len(normal) == 2 for normal in backbone["normals"])
        assert all(len(bound) == 2 for bound in backbone["offset_bounds"])


def test_every_backbone_value_is_finite(generated):
    train_dir, _, _ = generated

    for config in load_configs(train_dir):
        backbone = config["backbone"]
        for key in ("points", "normals", "offset_bounds"):
            assert np.isfinite(np.asarray(backbone[key], dtype=float)).all(), key
        assert np.isfinite(backbone["cumulative_distances"]).all()


def test_the_offset_bounds_bracket_zero(generated):
    train_dir, _, _ = generated

    for config in load_configs(train_dir):
        for low, high in config["backbone"]["offset_bounds"]:
            assert low <= 0.0 <= high


def test_the_heightmap_matches_the_declared_shape(generated):
    train_dir, _, _ = generated

    for path in sorted(train_dir.glob("*_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        heightmap = np.load(train_dir / f"{config['name']}_heightmap.npy")
        assert list(heightmap.shape) == config["heightmap_shape"]


def test_the_train_and_eval_routes_are_disjoint(generated):
    train_dir, eval_dir, _ = generated

    def endpoints(directory):
        return {
            (tuple(config["start"]), tuple(config["goal"])) for config in load_configs(directory)
        }

    assert not endpoints(train_dir) & endpoints(eval_dir)


def test_the_train_summary_recommends_working_dimensions(generated):
    train_dir, _, dimensions = generated

    summary = json.loads((train_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["dimensions_allowed"] == dimensions
    assert dimensions == sorted(dimensions)
    assert len(dimensions) == len(set(dimensions))


def test_the_eval_summary_omits_the_recommendation(generated):
    _, eval_dir, _ = generated

    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))

    assert "dimensions_allowed" not in summary


def test_each_summary_indexes_its_own_set(generated):
    train_dir, eval_dir, _ = generated

    for directory in (train_dir, eval_dir):
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        names = {config["name"] for config in load_configs(directory)}
        assert set(summary["instances"]) == names
        assert summary["total_instances"] == len(names)


def test_generation_is_reproducible_for_a_seed(tmp_path, terrain, small_config):
    first = tmp_path / "first"
    second = tmp_path / "second"

    prepare_instances.generate(terrain, small_config, first / "train", first / "eval")
    prepare_instances.generate(terrain, small_config, second / "train", second / "eval")

    assert load_configs(first / "train") == load_configs(second / "train")
    assert load_configs(first / "eval") == load_configs(second / "eval")


def test_the_cli_generates_a_set_from_a_source_file(tmp_path, terrain, small_config):
    source = tmp_path / "terrain.npy"
    np.save(source, terrain)

    exit_code = prepare_instances.main(
        [
            "--source",
            str(source),
            "--train-dir",
            str(tmp_path / "train"),
            "--eval-dir",
            str(tmp_path / "eval"),
            "--n-train",
            str(small_config.n_train),
            "--n-eval",
            str(small_config.n_eval),
            "--resolution",
            str(small_config.target_resolution[0]),
            str(small_config.target_resolution[1]),
            "--min-distance-fraction",
            str(small_config.min_distance_fraction),
            "--oversample-factor",
            str(small_config.oversample_factor),
            "--n-buckets",
            str(small_config.n_buckets),
            "--n-dimensions-allowed",
            str(small_config.n_dimensions_allowed),
            "--backbone-step",
            str(small_config.backbone_step),
            "--seed",
            str(small_config.seed),
        ]
    )

    assert exit_code == 0
    assert len(load_configs(tmp_path / "train")) == small_config.n_train
    assert (tmp_path / "eval" / "summary.json").is_file()
