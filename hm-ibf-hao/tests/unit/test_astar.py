"""Unit tests for :mod:`preprocessing.astar`."""

from __future__ import annotations

import numpy as np
import pytest
from preprocessing import astar


def test_downsampling_averages_over_blocks():
    heightmap = np.arange(16, dtype=float).reshape(4, 4)

    downsampled = astar.downsample(heightmap, 2, 2)

    assert downsampled.shape == (2, 2)
    assert downsampled[0, 0] == pytest.approx(np.mean([0, 1, 4, 5]))


def test_downsampling_keeps_a_map_that_is_already_small_enough():
    heightmap = np.zeros((3, 3))

    assert astar.downsample(heightmap, 10, 10).shape == (3, 3)


def test_downsampling_rejects_a_non_two_dimensional_map():
    with pytest.raises(ValueError, match="2-D"):
        astar.downsample(np.zeros(5), 2, 2)


def test_downsampling_rejects_a_non_positive_target():
    with pytest.raises(ValueError, match="positive"):
        astar.downsample(np.zeros((4, 4)), 0, 2)


def test_neighbours_of_a_corner_stay_in_bounds():
    found = set(astar.neighbours((0, 0), (3, 3)))

    assert found == {(0, 1), (1, 0), (1, 1)}


def test_an_interior_cell_has_eight_neighbours():
    assert len(set(astar.neighbours((1, 1), (3, 3)))) == 8


def test_the_euclidean_distance_is_symmetric():
    assert astar.euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)
    assert astar.euclidean_distance((3, 4), (0, 0)) == pytest.approx(5.0)


def test_climbing_and_descending_cost_the_same():
    heightmap = np.array([[0.0, 10.0], [0.0, 0.0]])

    up = astar.step_cost(heightmap, (0, 0), (0, 1))
    down = astar.step_cost(heightmap, (0, 1), (0, 0))

    assert up == down == pytest.approx(11.0)


def test_a_route_starts_at_the_start_and_ends_at_the_goal():
    heightmap = np.zeros((10, 10))

    route = astar.find_route(heightmap, (0, 0), (9, 9))

    assert route[0] == (0, 0)
    assert route[-1] == (9, 9)


def test_a_route_only_takes_adjacent_steps():
    heightmap = np.zeros((8, 8))

    route = astar.find_route(heightmap, (0, 0), (7, 7))

    for current, following in zip(route, route[1:], strict=False):
        assert max(abs(current[0] - following[0]), abs(current[1] - following[1])) == 1


def test_a_route_to_the_start_is_a_single_cell():
    assert astar.find_route(np.zeros((4, 4)), (2, 2), (2, 2)) == [(2, 2)]


def test_a_route_avoids_an_expensive_ridge():
    heightmap = np.zeros((9, 9))
    heightmap[4, :7] = 500.0

    route = astar.find_route(heightmap, (0, 3), (8, 3))

    crossings = [cell for cell in route if cell[0] == 4]
    assert all(cell[1] >= 7 for cell in crossings), f"route crossed the ridge: {route}"


def test_routing_from_outside_the_map_is_rejected():
    with pytest.raises(ValueError, match="start"):
        astar.find_route(np.zeros((4, 4)), (-1, 0), (2, 2))
    with pytest.raises(ValueError, match="goal"):
        astar.find_route(np.zeros((4, 4)), (0, 0), (9, 9))


def test_the_length_of_a_diagonal_route_is_its_arc_length():
    route = [(0, 0), (1, 1), (2, 2)]

    assert astar.route_length(route) == pytest.approx(2 * np.sqrt(2))


def test_a_route_of_fewer_than_two_cells_has_no_length():
    assert astar.route_length([]) == 0.0
    assert astar.route_length([(1, 1)]) == 0.0
