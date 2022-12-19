from typing import Tuple
from numba import prange, njit
import numpy as np

from pymatchseries.utils import DenseArrayType


class BilinearInterpolation2D:

    def __init__(
        self,
        image: DenseArrayType,
    ) -> None:
        self.image = image

    def evaluate(self, coordinates: DenseArrayType) -> DenseArrayType:
        pass

    def evaluate_gradient(self, coordinates: DenseArrayType) -> DenseArrayType:
        pass


@njit(parallel=True)
def interpolate_cpu(
    image: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate image at non-integer coordinates with linear interpolation
    """
    result = np.empty(coordinates.shape[:2], dtype=np.float32)
    rows = coordinates.shape[0]
    columns = coordinates.shape[1]
    for row in prange(rows):
        temp_weights = np.empty((2, 2), np.float32)
        for column in range(columns):
            y = coordinates[row, column, 0]
            x = coordinates[row, column, 1]
            _, y0, wy = _get_interpolation_parameters(y, image.shape[0])
            _, x0, wx = _get_interpolation_parameters(x, image.shape[1])
            sample = image[y0: y0 + 2, x0: x0 + 2]
            one_minus_wx = 1 - wx
            one_minus_wy = 1 - wy
            temp_weights[0, 0] = one_minus_wx * one_minus_wy
            temp_weights[0, 1] = one_minus_wy * wx
            temp_weights[1, 0] = wy * one_minus_wx
            temp_weights[1, 1] = wy * wx
            result[row, column] = np.sum(sample * temp_weights)
    return result


@njit
def interpolate_gradient_cpu_1core(
    image: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate image gradient at non-integer coordinates with linear interpolation
    """
    result = np.zeros(coordinates.shape, dtype=np.float32)
    rows = coordinates.shape[0]
    columns = coordinates.shape[1]

    for row in prange(rows):
        for column in range(columns):
            y = coordinates[row, column, 0]
            x = coordinates[row, column, 1]
            valid_y, y0, wy = _get_interpolation_parameters(y, image.shape[0])
            valid_x, x0, wx = _get_interpolation_parameters(x, image.shape[1])

            one_minus_wx = 1. - wx
            one_minus_wy = 1. - wy
            y1 = y0 + 1
            x1 = x0 + 1

            if valid_y:
                result[row, column, 0] = (
                    (image[y1, x0] - image[y0, x0]) * one_minus_wx +
                    (image[y1, x1] - image[y0, x1]) * wx
                )

            if valid_x:
                result[row, column, 1] = (
                    (image[y0, x1] - image[y0, x0]) * one_minus_wy +
                    (image[y1, x1] - image[y1, x0]) * wy
                )

    return result


@njit(parallel=True)
def interpolate_gradient_cpu(
    image: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate image gradient at non-integer coordinates with linear interpolation
    """
    result = np.zeros(coordinates.shape, dtype=np.float32)
    rows = coordinates.shape[0]
    columns = coordinates.shape[1]
    for row in prange(rows):
        for column in range(columns):
            y = coordinates[row, column, 0]
            x = coordinates[row, column, 1]
            valid_y, y0, wy = _get_interpolation_parameters(y, image.shape[0])
            valid_x, x0, wx = _get_interpolation_parameters(x, image.shape[1])

            one_minus_wx = 1. - wx
            one_minus_wy = 1. - wy
            y1 = y0 + 1
            x1 = x0 + 1

            if valid_y:
                result[row, column, 0] = (
                    (image[y1, x0] - image[y0, x0]) * one_minus_wx +
                    (image[y1, x1] - image[y0, x1]) * wx
                )

            if valid_x:
                result[row, column, 1] = (
                    (image[y0, x1] - image[y0, x0]) * one_minus_wy +
                    (image[y1, x1] - image[y1, x0]) * wy
                )

    return result


@njit
def _get_interpolation_parameters(
    coordinate: float,
    axis_size: int,
) -> Tuple[bool, int, float]:
    if coordinate >= 0 and coordinate < axis_size - 1:
        is_valid = True
        reference_gridpoint = int(coordinate)
        weight = coordinate - reference_gridpoint
    elif coordinate < 0:
        is_valid = False
        reference_gridpoint = 0
        weight = 0.
    elif coordinate > axis_size - 1:
        is_valid = False
        reference_gridpoint = axis_size - 2
        weight = 1.
    elif coordinate == axis_size - 1:
        is_valid = True
        reference_gridpoint = axis_size - 2
        weight = 1.
    return is_valid, reference_gridpoint, weight
