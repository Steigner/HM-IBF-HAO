"""Unit tests for :mod:`preprocessing.instance`."""

from __future__ import annotations

import json

import numpy as np
import pytest
from preprocessing import instance
from preprocessing.config import PreprocessingConfig
from preprocessing.sampling import Candidate


def make_candidate() -> Candidate:
    """A straight candidate route long enough to build a backbone from."""
    route = [(i, 10) for i in range(20)]
    return Candidate(start=route[0], goal=route[-1], route=route, length=19.0, bucket=0)


def test_the_payload_carries_every_field_the_rust_loader_reads(small_config):
    payload = instance.build_config(make_candidate(), (40, 60), "route_train_00", small_config)

    for key in (
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
    ):
        assert key in payload, key
    for key in ("points", "cumulative_distances", "total_length", "normals", "offset_bounds"):
        assert key in payload["backbone"], key


def test_the_payload_records_the_cost_parameters(small_config):
    config = PreprocessingConfig(**{**small_config.__dict__, "tunnel_factor": 9.0, "tau": 0.75})

    payload = instance.build_config(make_candidate(), (40, 60), "route_train_00", config)

    assert payload["tunnel_factor"] == 9.0
    assert payload["tau"] == 0.75


def test_the_payload_is_json_serializable(small_config):
    payload = instance.build_config(make_candidate(), (40, 60), "route_train_00", small_config)

    assert json.loads(json.dumps(payload))["name"] == "route_train_00"


def test_the_natural_dimension_never_drops_below_two(small_config):
    payload = instance.build_config(make_candidate(), (40, 60), "straight", small_config)

    assert payload["natural_dimension"] >= 2


def test_writing_an_instance_produces_the_expected_file_pair(tmp_path, small_config):
    payload = instance.build_config(make_candidate(), (40, 60), "route_train_00", small_config)
    heightmap = np.zeros((40, 60))

    instance.write_instance(payload, heightmap, tmp_path)

    assert (tmp_path / "route_train_00_config.json").is_file()
    assert (tmp_path / "route_train_00_heightmap.npy").is_file()


def test_the_written_heightmap_is_float32(tmp_path, small_config):
    payload = instance.build_config(make_candidate(), (40, 60), "route_train_00", small_config)

    instance.write_instance(payload, np.zeros((40, 60)), tmp_path)

    stored = np.load(tmp_path / "route_train_00_heightmap.npy")
    assert stored.dtype == np.float32
    assert stored.shape == (40, 60)


def test_writing_an_instance_creates_a_missing_directory(tmp_path, small_config):
    payload = instance.build_config(make_candidate(), (40, 60), "route_train_00", small_config)

    instance.write_instance(payload, np.zeros((40, 60)), tmp_path / "nested" / "deeper")

    assert (tmp_path / "nested" / "deeper" / "route_train_00_config.json").is_file()


def test_the_summary_indexes_every_instance(tmp_path, small_config):
    payloads = [
        instance.build_config(make_candidate(), (40, 60), f"route_train_{i:02d}", small_config)
        for i in range(3)
    ]

    path = instance.write_summary(tmp_path, payloads, seed=7)

    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["total_instances"] == 3
    assert summary["seed"] == 7
    assert len(summary["instances"]) == 3
    assert set(summary["backbone_lengths"]) == set(summary["instances"])
    assert set(summary["natural_dimensions"]) == set(summary["instances"])


def test_the_summary_only_carries_dimensions_when_given(tmp_path, small_config):
    payloads = [instance.build_config(make_candidate(), (40, 60), "a", small_config)]

    without = json.loads(
        instance.write_summary(tmp_path / "a", payloads, 1).read_text(encoding="utf-8")
    )
    with_dims = json.loads(
        instance.write_summary(tmp_path / "b", payloads, 1, [3, 5]).read_text(encoding="utf-8")
    )

    assert "dimensions_allowed" not in without
    assert with_dims["dimensions_allowed"] == [3, 5]


def test_a_degenerate_route_is_rejected(small_config):
    candidate = Candidate(start=(1, 1), goal=(1, 1), route=[(1, 1)], length=0.0, bucket=0)

    with pytest.raises(ValueError, match="at least two"):
        instance.build_config(candidate, (40, 60), "degenerate", small_config)
