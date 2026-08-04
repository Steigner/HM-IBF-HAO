"""End-to-end generation of the benchmark instance set from the named terrain maps.

Drives :mod:`preprocessing.prepare_instances` in benchmark mode — the path that regenerates
the checked-in `instances/` — against a pre-populated cache, so the test stays offline while
still exercising the download, routing, backbone and summary steps in one go.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from preprocessing import benchmark, prepare_instances
from preprocessing.maps import map_by_name


@pytest.fixture
def catalog(terrain):
    """Two catalogued maps shrunk to the synthetic terrain's shape."""
    base = map_by_name("austria")
    return (
        replace(
            base,
            name="austria",
            filename="austria_tiny.npy",
            source_shape=terrain.shape,
            resolution=terrain.shape,
            start=(5, 5),
            goal=(35, 55),
        ),
        replace(
            base,
            name="slovenia",
            filename="slovenia_tiny.npy",
            source_shape=terrain.shape,
            resolution=terrain.shape,
            start=(35, 5),
            goal=(5, 55),
        ),
    )


@pytest.fixture
def generated(tmp_path, terrain, catalog, small_config, monkeypatch):
    """Generate the benchmark set once from a pre-populated cache."""
    cache = tmp_path / "cache"
    cache.mkdir()
    for entry in catalog:
        np.save(cache / entry.filename, terrain)

    lookup = {entry.name: entry for entry in catalog}
    monkeypatch.setattr(benchmark, "BENCHMARK_MAPS", catalog)
    monkeypatch.setattr(benchmark, "map_by_name", lambda name: lookup[name])

    output_dir = tmp_path / "instances"
    exit_code = prepare_instances.main(
        [
            "--output-dir",
            str(output_dir),
            "--maps",
            *[entry.name for entry in catalog],
            "--cache-dir",
            str(cache),
            "--epsilon",
            str(small_config.epsilon),
            "--backbone-step",
            str(small_config.backbone_step),
            "--n-dimensions-allowed",
            str(small_config.n_dimensions_allowed),
        ]
    )

    assert exit_code == 0
    return output_dir


def load_configs(directory):
    """Read every instance config in a directory, in name order."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*_config.json"))
    ]


def test_one_instance_is_written_per_catalogued_map(generated, catalog):
    configs = load_configs(generated)

    assert len(configs) == len(catalog)
    assert {config["name"].split("_eps")[0] for config in configs} == {
        entry.name for entry in catalog
    }


def test_every_instance_is_paired_with_its_heightmap(generated):
    for config in load_configs(generated):
        assert (generated / f"{config['name']}_heightmap.npy").is_file()


def test_every_route_runs_between_the_catalogued_endpoints(generated, catalog):
    lookup = {entry.name: entry for entry in catalog}

    for config in load_configs(generated):
        entry = lookup[config["name"].split("_eps")[0]]
        assert config["path_astar"][0] == list(entry.start)
        assert config["path_astar"][-1] == list(entry.goal)


def test_the_summary_carries_the_dimension_recommendation(generated):
    summary = json.loads((generated / "summary.json").read_text(encoding="utf-8"))
    natural = [config["natural_dimension"] for config in load_configs(generated)]

    assert summary["dimensions_allowed"] == sorted(set(natural))
    assert set(summary["instances"]) == {config["name"] for config in load_configs(generated)}


def test_generation_is_reproducible(tmp_path, terrain, catalog, small_config, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    for entry in catalog:
        np.save(cache / entry.filename, terrain)
    monkeypatch.setattr(benchmark, "BENCHMARK_MAPS", catalog)

    first = tmp_path / "first"
    second = tmp_path / "second"
    benchmark.generate_benchmark(first, small_config, cache_dir=cache)
    benchmark.generate_benchmark(second, small_config, cache_dir=cache)

    assert load_configs(first) == load_configs(second)
