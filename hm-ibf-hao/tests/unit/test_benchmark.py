"""Unit tests for :mod:`preprocessing.benchmark`."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from preprocessing import benchmark
from preprocessing.maps import map_by_name


@pytest.fixture
def entry():
    """A catalog entry shrunk to the synthetic `terrain` fixture's shape."""
    return replace(
        map_by_name("austria"),
        filename="tiny.npy",
        source_shape=(40, 60),
        resolution=(40, 60),
        start=(5, 5),
        goal=(35, 55),
    )


def test_a_map_is_routed_between_its_catalog_endpoints(entry, terrain, small_config):
    payload = benchmark.build_map_instance(entry, terrain, small_config)

    assert payload["start"] == list(entry.start)
    assert payload["goal"] == list(entry.goal)
    assert payload["path_astar"][0] == list(entry.start)
    assert payload["path_astar"][-1] == list(entry.goal)


def test_the_instance_name_records_the_tolerance(entry, terrain, small_config):
    payload = benchmark.build_map_instance(entry, terrain, small_config)

    assert payload["name"] == f"{entry.name}_eps{small_config.epsilon}"


def test_endpoints_outside_the_heightmap_are_rejected(entry, terrain, small_config):
    outside = replace(entry, goal=(1000, 1000))

    with pytest.raises(ValueError, match="outside"):
        benchmark.build_map_instance(outside, terrain, small_config)


def test_the_set_is_written_with_its_summary(tmp_path, entry, terrain, small_config, monkeypatch):
    source = tmp_path / entry.filename
    np.save(source, terrain)
    monkeypatch.setattr(benchmark, "BENCHMARK_MAPS", (entry,))
    monkeypatch.setattr(benchmark, "ensure_heightmap", lambda *_args: source)

    dimensions = benchmark.generate_benchmark(tmp_path / "instances", small_config)

    summary = json.loads((tmp_path / "instances" / "summary.json").read_text(encoding="utf-8"))
    assert summary["dimensions_allowed"] == dimensions
    assert summary["total_instances"] == 1
    name = summary["instances"][0]
    assert (tmp_path / "instances" / f"{name}_config.json").is_file()
    assert (tmp_path / "instances" / f"{name}_heightmap.npy").is_file()


def test_the_recommendation_is_the_pool_of_natural_dimensions(
    tmp_path, entry, terrain, small_config, monkeypatch
):
    # With fewer maps than `n_dimensions_allowed`, the Douglas-Peucker dimensions of the
    # generated routes *are* the recommendation - no representative subset is taken.
    source = tmp_path / entry.filename
    np.save(source, terrain)
    monkeypatch.setattr(benchmark, "BENCHMARK_MAPS", (entry,))
    monkeypatch.setattr(benchmark, "ensure_heightmap", lambda *_args: source)

    dimensions = benchmark.generate_benchmark(tmp_path / "instances", small_config)

    configs = list((tmp_path / "instances").glob("*_config.json"))
    pool = sorted(
        {json.loads(path.read_text(encoding="utf-8"))["natural_dimension"] for path in configs}
    )
    assert dimensions == pool


def test_an_unknown_region_name_is_rejected(tmp_path, small_config):
    with pytest.raises(KeyError, match="atlantis"):
        benchmark.generate_benchmark(tmp_path, small_config, ["atlantis"])
