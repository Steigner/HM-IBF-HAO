"""Unit tests for :mod:`preprocessing.dimensions`."""

from __future__ import annotations

import pytest
from preprocessing import dimensions


def test_the_selection_is_sorted_and_deduplicated():
    selected = dimensions.select_dimensions_allowed([5, 9, 12, 20, 33, 41], 5)

    assert selected == sorted(selected)
    assert len(selected) == len(set(selected))


def test_the_selection_spans_the_pool():
    pool = [4, 8, 15, 16, 23, 42]

    selected = dimensions.select_dimensions_allowed(pool, 5)

    assert min(selected) == min(pool)
    assert max(selected) == max(pool)


def test_the_selection_includes_the_median():
    pool = [1, 2, 3, 100, 200]

    selected = dimensions.select_dimensions_allowed(pool, 5)

    assert 3 in selected


def test_a_uniform_pool_collapses_to_one_value():
    assert dimensions.select_dimensions_allowed([7, 7, 7, 7], 5) == [7]


def test_asking_for_one_value_yields_the_minimum_and_median():
    selected = dimensions.select_dimensions_allowed([2, 4, 6], 1)

    assert selected == [2, 4]


def test_the_selection_never_exceeds_the_requested_count_by_construction():
    pool = list(range(1, 50))

    selected = dimensions.select_dimensions_allowed(pool, 5)

    assert len(selected) <= 5


def test_every_selected_value_comes_from_the_pool_or_is_its_median():
    pool = [3, 11, 19, 27, 35]

    selected = dimensions.select_dimensions_allowed(pool, 4)

    assert all(value in pool for value in selected)


def test_an_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        dimensions.select_dimensions_allowed([], 5)


def test_a_non_positive_count_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        dimensions.select_dimensions_allowed([1, 2, 3], 0)


def test_a_pool_no_larger_than_the_count_is_taken_verbatim():
    # The three benchmark maps: their Douglas-Peucker dimensions *are* the recommendation.
    assert dimensions.select_dimensions_allowed([68, 53, 60], 5) == [53, 60, 68]


def test_the_toml_setting_is_formatted_for_pasting():
    rendered = dimensions.format_toml_setting([53, 60, 68])

    assert rendered == "dimensions_allowed = [53, 60, 68]"


def test_the_toml_setting_handles_a_single_dimension():
    assert dimensions.format_toml_setting([5]) == "dimensions_allowed = [5]"
