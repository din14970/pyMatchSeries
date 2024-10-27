from __future__ import annotations

from pathlib import Path
from typing import Union

from pydantic.dataclasses import dataclass

Pathlike = Union[Path, str]


class _DataclassConfig:
    validate_assignment = True


@dataclass(config=_DataclassConfig)
class Regularization:
    parameter_value_at_start: float = 0.1
    factor_level: float = 1.0
    factor_stage: float = 0.1


@dataclass(config=_DataclassConfig)
class ObjectiveConfig:
    number_of_quadrature_points: int = 3
    cache_derivative_of_regularizer: bool = True


@dataclass(config=_DataclassConfig)
class SolverConfig:
    max_iterations: int = 500
    stop_epsilon: float = 1e-6
    start_step: float = 1.0
    show_progress: bool = True


@dataclass(config=_DataclassConfig)
class IOConfig:
    store_each_stage: bool = True
    store_each_image_comparison: bool = True
    store_each_level: bool = False


@dataclass(config=_DataclassConfig)
class JNRRConfig:
    device: str = "auto"
    n_levels: int = 3
    n_stages: int = 2
    reference_update_function: str = "median"
    regularization: Regularization = Regularization()
    objective: ObjectiveConfig = ObjectiveConfig()
    solver: SolverConfig = SolverConfig()
    io: IOConfig = IOConfig()

    @classmethod
    def default(cls) -> JNRRConfig:
        return JNRRConfig()
