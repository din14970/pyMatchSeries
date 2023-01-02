from __future__ import annotations
from typing import Optional, Any, Mapping, Iterator, Callable
import dask.array as da
from tqdm.auto import tqdm
from pathlib import Path

from dataclasses import dataclass as classic_dataclass
from pydantic.dataclasses import dataclass

from hyperspy.signals import Signal2D, ComplexSignal2D

from pymatchseries.utils import (
    DenseArrayType, create_image_pyramid, resize_image_stack, map_coordinates,
    displacement_to_coordinates, to_host, to_device, mean, median, get_dispatcher,
)
from pymatchseries.implementation.objective_functions import RegistrationObjectiveFunction
from pymatchseries.implementation.solvers import root_gauss_newton


class _DataclassConfig:
    validate_assignment = True


@dataclass(config=_DataclassConfig)
class Regularization:
    constant_start: float = 0.1
    factor_level: float = 1.
    factor_stage: float = 0.1


@dataclass(config=_DataclassConfig)
class ObjectiveConfig:
    number_of_quadrature_points: int = 3
    cache_derivative_of_regularizer: bool = True


@dataclass(config=_DataclassConfig)
class SolverConfig:
    max_iterations: int = 50
    stop_epsilon: float = 0.
    start_step: float = 1.
    show_progress: bool = True


@dataclass(config=_DataclassConfig)
class IOConfig:
    store_each_stage: bool = True
    store_each_image_comparison: bool = True
    store_each_level: bool = False


@dataclass(config=_DataclassConfig)
class JNNRConfig:
    device: str = "auto"
    n_levels: int = 3
    n_stages: int = 2
    reference_update_function: str = "median"
    regularization: Regularization = Regularization()
    objective: ObjectiveConfig = ObjectiveConfig()
    solver: SolverConfig = SolverConfig()
    io: IOConfig = IOConfig()


@classic_dataclass
class JNNRState:
    images: Signal2D
    reference_image: Optional[DenseArrayType] = None
    deformations: Optional[ComplexSignal2D] = None
    completed_stages: int = 0

    @classmethod
    def load(cls, filepath: str = "saved_jnnr_calculation") -> JNNRState:
        pass

    def save(self) -> None:
        pass


class JNRR:

    def __init__(
        self,
        images: Signal2D,
    ) -> None:
        self._validate_images(images)
        self.__config = JNNRConfig()
        self.__state = JNNRState(images)

    @classmethod
    def load(
        cls,
        filepath: Path,
    ) -> JNRR:
        pass

    def save(self) -> None:
        pass

    @property
    def images(self) -> Signal2D:
        return self.state.images

    @property
    def state(self) -> JNNRState:
        return self.__state

    def run(self) -> None:
        L = self.config.regularization.constant_start
        n_stages = self.config.n_stages

        for stage in range(n_stages):
            displacements = []
            corrected_images = []

            progress = tqdm(total=self.number_of_images)

            with progress:
                progress.set_description(
                    f"Stage: {stage + 1}, Image: 0/{self.number_of_images}"
                )
                # Registration - finding all displacements
                for i, image in enumerate(self._get_image_iterator(
                    self.images,
                    device=self.config.device,
                )):
                    dp = get_dispatcher(image)
                    if self.state.reference_image is None:
                        self.state.reference_image = image
                        displacement = dp.zeros((2, *image.shape), dtype=image.dtype)
                        displacements.append(self._displacement_to_complex(displacement))
                        corrected_images.append(image)
                        progress.update(n=1)
                        continue

                    displacement = self._get_multilevel_displacement_field(
                        image,
                        self.state.reference_image,
                        regularization_constant=L,
                        configuration=self.config,
                    )
                    displacements.append(self._displacement_to_complex(displacement))

                    corrected_image = self._apply_displacement(image, displacement)
                    corrected_images.append(corrected_image)

                    progress.set_description(
                        f"Stage: {stage + 1}, Image: {i + 1}/{self.number_of_images}"
                    )
                    progress.update(n=1)

                # Bias correction??

                self.state.reference_image = self._aggregate_stack(
                    dp.stack(corrected_images),
                )
                self.state.deformations = ComplexSignal2D(dp.stack(corrected_images))
                self.state.completed_stages = stage + 1
                L *= self.config.regularization.factor_stage

    @classmethod
    def _displacement_to_complex(cls, displacement: DenseArrayType) -> DenseArrayType:
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
    def config(self) -> JNNRConfig:
        return self.__config

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
    def _validate_images(cls, images: Signal2D) -> None:
        if not isinstance(images, Signal2D):
            raise ValueError("Images must be a HyperSpy Signal2D object.")
        if images.axes_manager.navigation_dimension != 1:
            raise ValueError("Navigation dimension must be one dimensional.")

    @classmethod
    def _get_image_iterator(
        cls,
        images: Signal2D,
        device: Optional[str] = None,
    ) -> Iterator[DenseArrayType]:
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
        image: DenseArrayType,
        displacement: DenseArrayType,
    ) -> DenseArrayType:
        return map_coordinates(
            image,
            displacement_to_coordinates(displacement),
        )

    @classmethod
    def _get_multilevel_displacement_field(
        cls,
        image_deformed: DenseArrayType,
        image_reference: DenseArrayType,
        regularization_constant: float,
        configuration: JNNRConfig,
    ) -> DenseArrayType:
        """Get the displacement field between two images by progressively scaling"""
        n_levels = configuration.n_levels

        im_def_pyramid = create_image_pyramid(
            image_deformed,
            n_levels,
            downscale_factor=2.,
        )
        im_ref_pyramid = create_image_pyramid(
            image_reference,
            n_levels,
            downscale_factor=2.,
        )

        displacement = None

        progress = tqdm(total=n_levels, leave=False)

        with progress:
            progress.set_description(
                f"Level: 0/{n_levels}"
            )
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
        image_deformed: DenseArrayType,
        image_reference: DenseArrayType,
        regularization_constant: float,
        configuration: JNNRConfig,
        displacement_start: Optional[DenseArrayType] = None,
    ) -> DenseArrayType:
        """Compare two images and get the optimized displacement field"""
        image_shape = image_deformed.shape
        objective_configuration = configuration.objective
        objective = RegistrationObjectiveFunction(
            image_deformed,
            image_reference,
            regularization_constant,
            number_of_quadrature_points=objective_configuration.number_of_quadrature_points,
            cache_derivative_of_regularizer=objective_configuration.cache_derivative_of_regularizer,
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
