"""Shared fixtures for the preprocessing test suite."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest
from preprocessing.config import PreprocessingConfig


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random generator, so every test is reproducible."""
    return np.random.default_rng(20250803)


@pytest.fixture
def terrain() -> np.ndarray:
    """A small synthetic heightmap with a ridge running across it."""
    rows, columns = 40, 60
    row_grid, column_grid = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    ridge = 300.0 * np.exp(-(((row_grid - rows / 2) / 4.0) ** 2))
    slope = 2.0 * column_grid
    return (ridge + slope).astype(np.float64)


@pytest.fixture
def small_config() -> PreprocessingConfig:
    """A configuration small enough to run end to end inside a test."""
    return PreprocessingConfig(
        n_train=2,
        n_eval=2,
        target_resolution=(40, 60),
        min_distance_fraction=0.3,
        oversample_factor=3,
        n_buckets=2,
        n_dimensions_allowed=3,
        backbone_step=2.0,
        seed=7,
    )


@pytest.fixture
def instances_dir() -> Path:
    """The directory holding the instances shipped with the crate."""
    return Path(__file__).resolve().parent.parent / "instances"


@pytest.fixture
def training_params() -> dict:
    """The repository root's `params_training.conf`, parsed."""
    path = Path(__file__).resolve().parents[2] / "params_training.conf"
    return tomllib.loads(path.read_text(encoding="utf-8"))
