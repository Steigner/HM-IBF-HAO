"""Cross-component checks on the instances checked into the crate.

The Rust benchmark reads these files directly, so their shape is a contract between the
generator and the loader. These tests fail loudly if a regenerated set ever drifts from it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from preprocessing import dimensions, maps, simplify


def instance_names(instances_dir):
    """Every instance name in the shipped set, in name order."""
    return sorted(
        path.name.removesuffix("_config.json") for path in instances_dir.glob("*_config.json")
    )


@pytest.fixture
def configs(instances_dir):
    """Every shipped instance config, keyed by name."""
    return {
        name: json.loads((instances_dir / f"{name}_config.json").read_text(encoding="utf-8"))
        for name in instance_names(instances_dir)
    }


def test_the_crate_ships_instances(instances_dir, configs):
    assert instances_dir.is_dir()
    assert configs, "the crate must ship at least one instance"


def test_every_shipped_config_has_its_heightmap(instances_dir, configs):
    for name in configs:
        assert (instances_dir / f"{name}_heightmap.npy").is_file()


def test_every_config_name_matches_its_file_name(configs):
    for name, config in configs.items():
        assert config["name"] == name


def test_every_backbone_is_internally_consistent(configs):
    for name, config in configs.items():
        backbone = config["backbone"]
        count = len(backbone["points"])

        assert count >= 2, name
        assert len(backbone["normals"]) == count, name
        assert len(backbone["offset_bounds"]) == count, name
        assert len(backbone["cumulative_distances"]) == count, name
        assert backbone["total_length"] > 0.0, name


def test_the_cumulative_distances_ascend_to_the_total_length(configs):
    for name, config in configs.items():
        distances = np.asarray(config["backbone"]["cumulative_distances"])

        assert distances[0] == pytest.approx(0.0), name
        assert np.all(np.diff(distances) >= 0.0), name
        assert distances[-1] == pytest.approx(config["backbone"]["total_length"]), name


def test_every_backbone_normal_is_a_unit_vector(configs):
    for name, config in configs.items():
        normals = np.asarray(config["backbone"]["normals"], dtype=float)

        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0), name


def test_every_offset_bound_brackets_zero(configs):
    for name, config in configs.items():
        for low, high in config["backbone"]["offset_bounds"]:
            assert low <= 0.0 <= high, name


def test_displaced_backbone_points_stay_on_the_terrain(instances_dir, configs):
    for name, config in configs.items():
        heightmap = np.load(instances_dir / f"{name}_heightmap.npy")
        points = np.asarray(config["backbone"]["points"], dtype=float)
        normals = np.asarray(config["backbone"]["normals"], dtype=float)
        bounds = np.asarray(config["backbone"]["offset_bounds"], dtype=float)

        for point, normal, (low, high) in zip(points, normals, bounds, strict=True):
            for offset in (low, high):
                displaced = point + offset * normal
                assert -1e-6 <= displaced[0] <= heightmap.shape[0] - 1 + 1e-6, name
                assert -1e-6 <= displaced[1] <= heightmap.shape[1] - 1 + 1e-6, name


def test_the_heightmap_matches_the_declared_shape(instances_dir, configs):
    for name, config in configs.items():
        heightmap = np.load(instances_dir / f"{name}_heightmap.npy")

        assert list(heightmap.shape) == config["heightmap_shape"], name


def test_the_route_runs_from_the_start_to_the_goal(configs):
    for name, config in configs.items():
        route = config["path_astar"]

        assert route[0] == config["start"], name
        assert route[-1] == config["goal"], name


def test_the_route_only_takes_adjacent_steps(configs):
    for name, config in configs.items():
        route = config["path_astar"]

        for current, following in zip(route, route[1:], strict=False):
            step = max(abs(current[0] - following[0]), abs(current[1] - following[1]))
            assert step == 1, name


def test_the_simplified_route_is_derived_from_the_raw_route(configs):
    for name, config in configs.items():
        raw = [tuple(point) for point in config["path_astar"]]

        recomputed = simplify.simplify(raw, config["epsilon"])

        assert [list(point) for point in recomputed] == config["path_simplified"], name


def test_the_natural_dimension_matches_the_simplified_route(configs):
    for name, config in configs.items():
        simplified = [tuple(point) for point in config["path_simplified"]]

        assert config["natural_dimension"] == simplify.natural_dimension(simplified), name


def test_the_cost_parameters_are_usable(configs):
    for name, config in configs.items():
        assert config["curvature_radius"] > 0.0, name
        assert config["tunnel_factor"] >= 1.0, name
        assert config["gradient_factor"] >= 1.0, name
        assert config["gradient_change_limit"] > 0.0, name


def test_every_shipped_instance_comes_from_a_catalogued_map(configs):
    for name, config in configs.items():
        entry = maps.map_by_name(name.split("_eps")[0])

        assert config["start"] == list(entry.start), name
        assert config["goal"] == list(entry.goal), name
        assert config["heightmap_shape"] == list(entry.resolution), name
        assert name == maps.instance_name(entry, config["epsilon"]), name


def test_the_allowed_dimensions_are_the_recommendation_of_the_shipped_pool(
    configs, training_params
):
    # `dimensions_allowed` is not a free parameter: it is what the Douglas-Peucker
    # compression of the shipped routes recommends. Editing one without the other silently
    # makes the islands search at a resolution the terrain never asked for.
    pool = [config["natural_dimension"] for config in configs.values()]

    recommended = dimensions.select_dimensions_allowed(
        pool, training_params.get("n_dimensions_allowed", len(pool))
    )

    assert training_params["dimensions_allowed"] == recommended


def test_the_summary_indexes_the_shipped_set(instances_dir, configs):
    summary = json.loads((instances_dir / "summary.json").read_text(encoding="utf-8"))

    assert set(summary["instances"]) == set(configs)
    assert summary["total_instances"] == len(configs)
    for name, config in configs.items():
        assert summary["natural_dimensions"][name] == config["natural_dimension"]
        assert summary["backbone_lengths"][name] == pytest.approx(
            config["backbone"]["total_length"]
        )
