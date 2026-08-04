"""Assembly and serialization of a single benchmark instance."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing import backbone as backbone_module
from preprocessing import simplify
from preprocessing.config import PreprocessingConfig
from preprocessing.sampling import Candidate

LOGGER = logging.getLogger(__name__)


def build_config(
    candidate: Candidate,
    heightmap_shape: tuple[int, int],
    name: str,
    config: PreprocessingConfig,
) -> dict[str, Any]:
    """Assemble the config payload of one instance.

    Args:
        candidate: The routed start/goal pair the instance is built from.
        heightmap_shape: `(rows, columns)` of the downsampled heightmap.
        name: The instance name; also the stem of both its files.
        config: The generator configuration supplying the cost parameters.

    Returns:
        The payload, matching the `AlignmentConfig` the Rust benchmark reads.

    Raises:
        ValueError: If the candidate's route cannot form a backbone.
    """
    simplified = simplify.simplify(candidate.route, config.epsilon)
    backbone = backbone_module.build(
        candidate.route,
        heightmap_shape,
        config.cutting_plane_factor,
        config.backbone_step,
    )

    return {
        "name": name,
        "heightmap_shape": list(heightmap_shape),
        "start": list(candidate.start),
        "goal": list(candidate.goal),
        "path_astar": [list(point) for point in candidate.route],
        "path_simplified": [list(point) for point in simplified],
        "natural_dimension": simplify.natural_dimension(simplified),
        "backbone": backbone,
        "epsilon": config.epsilon,
        "cutting_plane_factor": config.cutting_plane_factor,
        "tunnel_factor": config.tunnel_factor,
        "gradient_factor": config.gradient_factor,
        "curvature_radius": config.curvature_radius,
        "gradient_change_limit": config.gradient_change_limit,
        "height_limit": config.height_limit,
        "tau": config.tau,
    }


def write_instance(
    instance_config: dict[str, Any],
    heightmap: np.ndarray,
    output_dir: Path,
) -> None:
    """Write one instance's config and heightmap to disk.

    Args:
        instance_config: The payload returned by :func:`build_config`.
        heightmap: The downsampled heightmap the backbone refers to.
        output_dir: Directory receiving both files; created if missing.

    Raises:
        OSError: If either file cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    name = instance_config["name"]

    np.save(output_dir / f"{name}_heightmap.npy", heightmap.astype(np.float32))
    (output_dir / f"{name}_config.json").write_text(
        json.dumps(instance_config, indent=2), encoding="utf-8"
    )

    LOGGER.info(
        "wrote %s: backbone of %d points, length %.2f, natural dimension %d",
        name,
        len(instance_config["backbone"]["points"]),
        instance_config["backbone"]["total_length"],
        instance_config["natural_dimension"],
    )


def write_summary(
    output_dir: Path,
    configs: list[dict[str, Any]],
    seed: int,
    dimensions_allowed: list[int] | None = None,
) -> Path:
    """Write the index of an instance set.

    Args:
        output_dir: Directory receiving `summary.json`; created if missing.
        configs: The payloads of every instance in the set.
        seed: The seed the set was generated with.
        dimensions_allowed: The recommended working dimensions, when known.

    Returns:
        The path of the written summary.

    Raises:
        OSError: If the file cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "instances": [config["name"] for config in configs],
        "backbone_lengths": {
            config["name"]: config["backbone"]["total_length"] for config in configs
        },
        "natural_dimensions": {config["name"]: config["natural_dimension"] for config in configs},
        "total_instances": len(configs),
        "seed": seed,
    }
    if dimensions_allowed is not None:
        summary["dimensions_allowed"] = dimensions_allowed

    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
