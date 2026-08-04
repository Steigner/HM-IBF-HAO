"""Unit tests for :mod:`preprocessing.maps`."""

from __future__ import annotations

import pytest
from preprocessing import maps


def test_the_catalog_holds_the_three_benchmark_regions():
    assert maps.MAP_NAMES == ("austria", "italia", "slovenia")
    assert len(maps.BENCHMARK_MAPS) == len(maps.MAP_NAMES)


def test_every_entry_declares_a_routable_endpoint_pair():
    for entry in maps.BENCHMARK_MAPS:
        rows, columns = entry.resolution
        for cell in (entry.start, entry.goal):
            assert 0 <= cell[0] < rows, entry.name
            assert 0 <= cell[1] < columns, entry.name
        assert entry.start != entry.goal, entry.name


def test_every_entry_declares_a_source_larger_than_its_resolution():
    for entry in maps.BENCHMARK_MAPS:
        assert entry.source_shape[0] >= entry.resolution[0], entry.name
        assert entry.source_shape[1] >= entry.resolution[1], entry.name


def test_a_map_is_looked_up_by_name():
    assert maps.map_by_name("italia").filename.startswith("italia_")


def test_an_unknown_map_name_is_rejected():
    with pytest.raises(KeyError, match="atlantis"):
        maps.map_by_name("atlantis")


def test_the_instance_name_carries_the_simplification_tolerance():
    entry = maps.map_by_name("austria")

    assert maps.instance_name(entry, 1.0) == "austria_eps1.0"
