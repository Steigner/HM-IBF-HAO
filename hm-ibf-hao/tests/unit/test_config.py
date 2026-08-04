"""Unit tests for :mod:`preprocessing.config`."""

from __future__ import annotations

import dataclasses

import pytest
from preprocessing.config import PreprocessingConfig


def test_the_defaults_are_valid():
    PreprocessingConfig().validate()


def test_the_candidate_pool_covers_both_sets():
    config = PreprocessingConfig(n_train=10, n_eval=20, oversample_factor=3)

    assert config.n_candidates == 90


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_train", 0, "at least one instance"),
        ("n_eval", 0, "at least one instance"),
        ("epsilon", 0.0, "epsilon"),
        ("backbone_step", -1.0, "backbone_step"),
        ("target_resolution", (1, 10), "target_resolution"),
        ("min_distance_fraction", 0.0, "min_distance_fraction"),
        ("min_distance_fraction", 1.0, "min_distance_fraction"),
        ("oversample_factor", 0, "oversample_factor"),
        ("n_buckets", 0, "n_buckets"),
        ("n_dimensions_allowed", 0, "n_dimensions_allowed"),
        ("curvature_radius", 0.0, "curvature_radius"),
        ("tau", 1.5, "tau"),
        ("tau", -0.1, "tau"),
    ],
)
def test_an_out_of_range_field_is_rejected(field, value, message):
    config = dataclasses.replace(PreprocessingConfig(), **{field: value})

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_the_configuration_is_immutable():
    config = PreprocessingConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 1
