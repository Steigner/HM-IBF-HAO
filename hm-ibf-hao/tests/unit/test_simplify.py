"""Unit tests for :mod:`preprocessing.simplify`."""

from __future__ import annotations

import pytest
from preprocessing import simplify


def test_the_distance_to_a_horizontal_line_is_the_vertical_offset():
    assert simplify.point_line_distance((1, 5), (0, 0), (10, 0)) == pytest.approx(5.0)


def test_the_distance_to_a_degenerate_segment_is_the_point_distance():
    assert simplify.point_line_distance((3, 4), (0, 0), (0, 0)) == pytest.approx(5.0)


def test_a_point_on_the_line_has_no_distance():
    assert simplify.point_line_distance((5, 0), (0, 0), (10, 0)) == pytest.approx(0.0)


def test_a_straight_polyline_collapses_to_its_endpoints():
    points = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]

    assert simplify.simplify(points, 0.5) == [(0, 0), (4, 0)]


def test_a_significant_deviation_is_kept():
    points = [(0, 0), (2, 5), (4, 0)]

    assert simplify.simplify(points, 1.0) == points


def test_a_deviation_below_the_tolerance_is_dropped():
    points = [(0, 0), (2, 1), (4, 0)]

    assert simplify.simplify(points, 2.0) == [(0, 0), (4, 0)]


def test_simplification_keeps_both_endpoints():
    points = [(i, (i % 3)) for i in range(30)]

    simplified = simplify.simplify(points, 0.5)

    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]


def test_simplification_is_a_subsequence_of_its_input():
    points = [(i, (i * i) % 7) for i in range(40)]

    simplified = simplify.simplify(points, 1.0)

    iterator = iter(points)
    assert all(point in iterator for point in simplified)


def test_a_short_polyline_is_returned_unchanged():
    assert simplify.simplify([], 1.0) == []
    assert simplify.simplify([(0, 0)], 1.0) == [(0, 0)]
    assert simplify.simplify([(0, 0), (1, 1)], 1.0) == [(0, 0), (1, 1)]


def test_a_non_positive_tolerance_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        simplify.simplify([(0, 0), (1, 1), (2, 0)], 0.0)


def test_a_smaller_tolerance_keeps_at_least_as_many_points():
    points = [(i, (i % 5)) for i in range(50)]

    fine = simplify.simplify(points, 0.5)
    coarse = simplify.simplify(points, 3.0)

    assert len(fine) >= len(coarse)


def test_a_long_route_does_not_exhaust_the_stack():
    # Deviations that shrink along the route make the algorithm split next to an endpoint
    # every time, so the recursion depth grows with the route rather than with its log. At
    # this length a recursive implementation would exceed the interpreter's stack limit.
    count = 2_500
    points = [(i, count - i if i % 2 else 0) for i in range(count)]

    simplified = simplify.simplify(points, 0.5)

    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]
    assert len(simplified) > 1_000, "the route must actually force a deep recursion"


def test_the_natural_dimension_excludes_both_endpoints():
    assert simplify.natural_dimension([(0, 0)] * 10) == 8


def test_the_natural_dimension_never_drops_below_two():
    assert simplify.natural_dimension([(0, 0), (1, 1)]) == 2
    assert simplify.natural_dimension([]) == 2
