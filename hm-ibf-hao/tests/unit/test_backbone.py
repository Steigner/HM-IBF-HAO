"""Unit tests for :mod:`preprocessing.backbone`."""

from __future__ import annotations

import numpy as np
import pytest
from preprocessing import backbone


def straight_route(length: int = 20) -> list[tuple[int, int]]:
    """A straight route running along the first axis."""
    return [(i, 10) for i in range(length)]


def test_a_backbone_is_resampled_at_equal_arc_lengths():
    result = backbone.build(straight_route(), (40, 40), 1.0, 1.0)

    distances = np.diff(result["cumulative_distances"])
    assert np.allclose(distances, distances[0])


def test_the_backbone_reports_the_route_length():
    result = backbone.build(straight_route(21), (40, 40), 1.0, 1.0)

    assert result["total_length"] == pytest.approx(20.0)


def test_a_smaller_step_yields_more_backbone_points():
    coarse = backbone.build(straight_route(), (40, 40), 1.0, 4.0)
    fine = backbone.build(straight_route(), (40, 40), 1.0, 0.5)

    assert len(fine["points"]) > len(coarse["points"])


def test_all_backbone_arrays_have_the_same_length():
    result = backbone.build(straight_route(), (40, 40), 1.0, 1.0)

    count = len(result["points"])
    assert len(result["normals"]) == count
    assert len(result["offset_bounds"]) == count
    assert len(result["cumulative_distances"]) == count


def test_the_backbone_endpoints_match_the_route_endpoints():
    route = straight_route()

    result = backbone.build(route, (40, 40), 1.0, 1.0)

    assert result["points"][0] == pytest.approx(list(route[0]))
    assert result["points"][-1] == pytest.approx(list(route[-1]))


def test_a_route_shorter_than_two_points_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        backbone.build([(0, 0)], (10, 10), 1.0, 1.0)


def test_a_route_of_zero_length_is_rejected():
    with pytest.raises(ValueError, match="no length"):
        backbone.build([(3, 3), (3, 3)], (10, 10), 1.0, 1.0)


def test_normals_are_unit_length_and_perpendicular_to_the_tangent():
    points = np.array([[float(i), float(i)] for i in range(10)])

    normals = backbone.compute_normals(points)

    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    tangent = points[5] - points[3]
    assert np.dot(normals[4], tangent) == pytest.approx(0.0, abs=1e-9)


def test_a_stationary_stretch_falls_back_to_a_default_normal():
    points = np.zeros((5, 2))

    normals = backbone.compute_normals(points)

    assert np.allclose(normals, np.array([0.0, 1.0]))


def test_normals_are_computed_for_a_two_point_backbone():
    points = np.array([[0.0, 0.0], [0.0, 4.0]])

    normals = backbone.compute_normals(points)

    assert normals.shape == (2, 2)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_offset_bounds_bracket_zero():
    result = backbone.build(straight_route(), (40, 40), 1.0, 1.0)

    for low, high in result["offset_bounds"]:
        assert low <= 0.0 <= high


def test_offset_bounds_never_leave_the_heightmap():
    shape = (40, 40)
    result = backbone.build(straight_route(), shape, 5.0, 1.0)

    points = np.asarray(result["points"])
    normals = np.asarray(result["normals"])
    for point, normal, (low, high) in zip(points, normals, result["offset_bounds"], strict=True):
        for offset in (low, high):
            displaced = point + offset * normal
            assert -1e-9 <= displaced[0] <= shape[0] - 1 + 1e-9
            assert -1e-9 <= displaced[1] <= shape[1] - 1 + 1e-9


def test_a_larger_cutting_plane_factor_widens_the_bounds():
    narrow = backbone.build(straight_route(), (400, 400), 1.0, 1.0)
    wide = backbone.build(straight_route(), (400, 400), 3.0, 1.0)

    assert wide["offset_bounds"][5][1] > narrow["offset_bounds"][5][1]


def test_a_backbone_on_the_edge_cannot_move_off_the_map():
    # The route runs along row 0, so one side of every normal points off the map and the
    # bound on that side has to collapse to zero.
    route = [(0, i) for i in range(20)]

    result = backbone.build(route, (40, 40), 5.0, 1.0)

    for low, high in result["offset_bounds"]:
        assert min(abs(low), abs(high)) == pytest.approx(0.0, abs=1e-9)
    assert any(abs(low) > 0.0 for low, _ in result["offset_bounds"]), (
        "the inward side must still be free to move"
    )
