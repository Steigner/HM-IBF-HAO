"""Tunable defaults of the instance generator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Every knob of the instance generator, with its default.

    The command line exposes each field as a flag; grouping them here keeps the CLI, the
    tests and the documentation reading from one source.

    Attributes:
        n_train: Number of training instances to generate.
        n_eval: Number of evaluation instances to generate.
        epsilon: Douglas-Peucker tolerance used to estimate the natural dimension.
        cutting_plane_factor: Multiplier applied to the perpendicular offset bounds.
        backbone_step: Arc length between consecutive backbone points, in pixels.
        target_resolution: `(rows, columns)` the source heightmap is downsampled to.
        min_distance_fraction: Smallest start-goal distance, as a fraction of the
            downsampled map's diagonal.
        oversample_factor: Candidate pool size, as a multiple of the instances needed.
        n_buckets: Number of route-length buckets the candidates are spread over.
        n_dimensions_allowed: Number of working dimensions to recommend.
        seed: Seed of the sampling and splitting RNG.
        tunnel_factor: Cost multiplier for segments above `height_limit`.
        gradient_factor: Cost multiplier for segments steeper than `gradient_change_limit`.
        curvature_radius: Smallest radius of curvature a route may have.
        gradient_change_limit: Largest unpenalized absolute gradient.
        height_limit: Elevation above which the tunnel penalty applies.
        tau: Clothoid asymmetry parameter in `[0, 1]`.
    """

    n_train: int = 30
    n_eval: int = 90
    epsilon: float = 1.0
    cutting_plane_factor: float = 1.0
    backbone_step: float = 1.0
    target_resolution: tuple[int, int] = (200, 400)
    min_distance_fraction: float = 0.15
    oversample_factor: int = 2
    n_buckets: int = 3
    n_dimensions_allowed: int = 5
    seed: int = 42
    tunnel_factor: float = 5.0
    gradient_factor: float = 2.0
    curvature_radius: float = 100.0
    gradient_change_limit: float = 0.08
    height_limit: float = 800.0
    tau: float = 0.4

    def validate(self) -> None:
        """Check that the configuration describes a generatable instance set.

        Raises:
            ValueError: If any field is outside its admissible range.
        """
        if self.n_train < 1 or self.n_eval < 1:
            raise ValueError(
                f"both sets need at least one instance, got train={self.n_train} eval={self.n_eval}"
            )
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.backbone_step <= 0.0:
            raise ValueError(f"backbone_step must be positive, got {self.backbone_step}")
        if min(self.target_resolution) < 2:
            raise ValueError(
                f"target_resolution must be at least 2x2, got {self.target_resolution}"
            )
        if not 0.0 < self.min_distance_fraction < 1.0:
            raise ValueError(
                f"min_distance_fraction must lie in (0, 1), got {self.min_distance_fraction}"
            )
        if self.oversample_factor < 1:
            raise ValueError(f"oversample_factor must be at least 1, got {self.oversample_factor}")
        if self.n_buckets < 1:
            raise ValueError(f"n_buckets must be at least 1, got {self.n_buckets}")
        if self.n_dimensions_allowed < 1:
            raise ValueError(
                f"n_dimensions_allowed must be at least 1, got {self.n_dimensions_allowed}"
            )
        if self.curvature_radius <= 0.0:
            raise ValueError(f"curvature_radius must be positive, got {self.curvature_radius}")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"tau must lie in [0, 1], got {self.tau}")

    @property
    def n_candidates(self) -> int:
        """Number of start/goal pairs sampled before the train/eval split.

        Returns:
            The candidate pool size.
        """
        return (self.n_train + self.n_eval) * self.oversample_factor
