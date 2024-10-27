from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Optional

import dask.array as da
from hyperspy.signals import ComplexSignal2D, Signal2D
from tqdm.auto import tqdm

from pymatchseries.implementation.configuration import JNRRConfig
from pymatchseries.implementation.objective_functions import (
    RegistrationObjectiveFunction,
)
from pymatchseries.implementation.solvers import root_gauss_newton
from pymatchseries.utils import (
    create_image_pyramid,
    displacement_to_coordinates,
    get_dispatcher,
    map_coordinates,
    mean,
    median,
    resize_image_stack,
    to_device,
    to_host,
)
from numpy.typing import NDArray


@dataclass
class JNRRResults:
    deformations: ComplexSignal2D
    corrected_images: Signal2D


@dataclass
class JNRRState:
    images: Signal2D
    config: JNRRConfig
    reference_image_index: int
    stage_index: int

    @classmethod
    def new(
        cls,
        images: Signal2D,
    ) -> JNRRState:
        return JNRRState(
            images=images,
            config=JNRRConfig.default(),
            reference_image_index=0,
            stage_index=0,
        )

    @classmethod
    def load(cls, filepath: Path) -> JNRRState:
        raise NotImplementedError()

    def save(self, filepath: Path) -> None:
        raise NotImplementedError()


class JNRRJob:
    """Joint non rigid registration API"""

    def __init__(
        self,
        state: JNRRState,
    ) -> None:
        self._state = state

    @classmethod
    def from_signal(
        cls,
        signal: Signal2D,
    ) -> JNRRJob:
        """Start a JNRR calculation from a Hyperspy Signal2D object"""
        cls._validate_signal(signal)
        state = JNRRState.new(signal)
        return JNRRJob(state)

    @classmethod
    def load(
        cls,
        filepath: Path,
    ) -> JNRRJob:
        state = JNRRState.load(filepath)
        return cls(state)

    def save(
        self,
        filepath: Path,
    ) -> None:
        self._state.save(filepath)

    @property
    def images(self) -> Signal2D:
        return self._state.images

    @cached_property
    def dispatcher(self) -> ModuleType:
        return get_dispatcher(self.images.data)

    def run(self) -> None:
        L = self.config.regularization.parameter_value_at_start
        n_stages = self.config.n_stages
        dp = self.dispatcher

        displacements = dp.zeros((2, *self.images.shape), dtype=dp.float32)
        corrected_images = dp.empty(self.images.shape, dtype=dp.float32)

        for stage in range(n_stages):

            progress = tqdm(total=self.number_of_images)

            image_pyramid_reference = list(create_image_pyramid(
                self.state.reference_image,
                n_levels,
                downscale_factor=2.0,
            ))

            # Registration - finding all displacements
            with progress:
                progress.set_description(
                    f"Stage: {stage + 1}, Image: 0/{self.number_of_images}"
                )
                for i, image in enumerate(
                    self._get_image_iterator(
                        self.images,
                        device=self.config.device,
                    )
                ):
                    if self.state.reference_image is None:
                        self.state.reference_image = image
                        corrected_images[i] = image
                        progress.update(n=1)
                        continue

                    im_def_pyramid = create_image_pyramid(
                        image_deformed,
                        n_levels,
                        downscale_factor=2.0,
                    )

                    displacements[:, i, ...] = self._get_multilevel_displacement_field(
                        image,
                        self.state.reference_image,
                        regularization_constant=L,
                        configuration=self.config,
                    )

                    corrected_images[i, ...] = self._apply_displacement(
                        image,
                        displacements[:, i, ...],
                    )

                    progress.set_description(
                        f"Stage: {stage + 1}, Image: {i + 1}/{self.number_of_images}"
                    )
                    progress.update(n=1)

            if stage == 0:
                L *= self.config.regularization.factor_stage

            # Bias correction step

            self.state.reference_image = self._aggregate_stack(corrected_images)
            self.state.completed_stages = stage + 1

        self.state.deformations = ComplexSignal2D(
            self._displacement_to_complex(displacements)
        )
        self.state.corrected_images = Signal2D(corrected_images)

    @classmethod
    def _displacement_to_complex(
        cls,
        displacement: NDArray,
    ) -> NDArray:
        return displacement[1] + 1j * displacement[0]

    @property
    def _aggregate_stack(self) -> Callable:
        if self.config.reference_update_function == "mean":
            return mean
        elif self.config.reference_update_function == "median":
            return median
        else:
            raise NotImplementedError("Unrecognized aggregation method.")

    @property
    def number_of_images(self) -> int:
        return self.images.axes_manager.navigation_size

    @property
    def config(self) -> JNRRConfig:
        return self._state.config

    def configure(self, options: Mapping[str, Any]) -> None:
        """Provide configuration in a dictionary using dot notation"""
        for key, value in options.items():
            split_key = key.split(".")
            obj = self.config
            for key_part in split_key:
                sub_obj = getattr(obj, key_part)
                if hasattr(sub_obj, "__dataclass_fields__"):
                    obj = sub_obj
            setattr(obj, key_part, value)

    @classmethod
    def _validate_signal(cls, signal: Signal2D) -> None:
        if not isinstance(signal, Signal2D):
            raise ValueError("Images must be a HyperSpy Signal2D object.")
        if signal.axes_manager.navigation_dimension != 1:
            raise ValueError("Navigation dimension must be one dimensional.")

    @classmethod
    def _get_image_iterator(
        cls,
        images: Signal2D,
        device: Optional[str] = None,
    ) -> Iterator[NDArray]:
        is_lazy = isinstance(images.data, da.Array)

        if device == "cpu":
            transfer = to_host
        elif device == "gpu":
            transfer = to_device
        else:

            def do_nothing(x):
                return x

            transfer = do_nothing

        for image in iter(images):
            if is_lazy:
                image.compute()
            yield transfer(image.data)

    @classmethod
    def _apply_displacement(
        cls,
        image: NDArray,
        displacement: NDArray,
    ) -> NDArray:
        return map_coordinates(
            image,
            displacement_to_coordinates(displacement),
        )

    @classmethod
    def _get_multilevel_displacement_field(
        cls,
        image_pyramid_deformed: Iterable[NDArray],
        image_pyramid_reference: Iterable[NDArray],
        regularization_constant: float,
        configuration: JNRRConfig,
    ) -> NDArray:
        """Get the displacement field between two images by progressively scaling"""
        n_levels = configuration.n_levels

        displacement = None

        progress = tqdm(total=n_levels, leave=False)

        with progress:
            progress.set_description(f"Level: 0/{n_levels}")
            for i, (im_def, im_ref) in enumerate(zip(im_def_pyramid, im_ref_pyramid)):
                if displacement is not None:
                    displacement = resize_image_stack(
                        image_stack=displacement,
                        new_size=im_def.shape,
                    )

                displacement = cls._get_displacement_field(
                    image_deformed=im_def,
                    image_reference=im_ref,
                    regularization_constant=regularization_constant,
                    configuration=configuration,
                    displacement_start=displacement,
                )

                regularization_constant *= configuration.regularization.factor_level

                progress.set_description(
                    f"Level: {i + 1}/{n_levels}, Size: {im_def.shape}"
                )
                progress.update(n=1)

        return displacement

    @classmethod
    def _get_displacement_field(
        cls,
        image_deformed: NDArray,
        image_reference: NDArray,
        regularization_constant: float,
        configuration: JNRRConfig,
        displacement_start: Optional[NDArray] = None,
    ) -> NDArray:
        """Compare two images and get the optimized displacement field"""
        image_shape = image_deformed.shape
        objective_configuration = configuration.objective
        n_quad_points = objective_configuration.number_of_quadrature_points
        cache_d_reg = objective_configuration.cache_derivative_of_regularizer
        objective = RegistrationObjectiveFunction(
            image_deformed,
            image_reference,
            regularization_constant,
            number_of_quadrature_points=n_quad_points,
            cache_derivative_of_regularizer=cache_d_reg,
        )

        dp = objective.dispatcher

        if displacement_start is not None:
            displacement_vector = displacement_start.ravel()
        else:
            displacement_vector = dp.zeros(2 * objective.number_of_nodes)

        solver_configuration = configuration.solver
        return root_gauss_newton(
            F=objective.evaluate_residual,
            x0=displacement_vector,
            DF=objective.evaluate_residual_gradient,
            max_iterations=solver_configuration.max_iterations,
            stop_epsilon=solver_configuration.stop_epsilon,
            start_step=solver_configuration.start_step,
            show_progress=solver_configuration.show_progress,
        ).reshape(2, *image_shape)
