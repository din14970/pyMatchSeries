from typing import Tuple
from numba import prange, njit
import numpy as np

from pymatchseries.utils import DenseArrayType, get_dispatcher, cp


class BilinearInterpolation2D:
    """Class to perform interpolation of an image on an arbitrary non-integer
    coordinate grid

    Parameters
    ----------
    image: (N, M) array of float32
        The array to use for interpolation
    """

    def __init__(
        self,
        image: DenseArrayType,
    ) -> None:
        self.image = image
        dispatcher = get_dispatcher(image)
        if dispatcher == np:
            self.evaluate_function = interpolate_cpu
            self.evaluate_gradient_function = interpolate_gradient_cpu
        elif dispatcher == cp:
            from pymatchseries.implementation.cuda_kernels import (
                interpolate_gpu,
                interpolate_gradient_gpu,
            )
            self.evaluate_function = interpolate_gpu
            self.evaluate_function = interpolate_gradient_gpu
        else:
            raise ValueError("Unexpected object type for image")

    def evaluate(self, coordinates: DenseArrayType) -> DenseArrayType:
        """Evaluate image at non-integer coordinates with linear interpolation

        Parameters
        ----------
        coordinates: (R, C, 2) array of float32
            The coordinates at which to interpolate. (R, C, 0) represent the y
            coordinates in the image, (R, C, 1) represent the x coordinate in
            the image

        Returns
        -------
        values: (R, C) array of float32
            The interpolated values for all R, C coordinates
        """
        return self.evaluate_function(self.image, coordinates)

    def evaluate_gradient(self, coordinates: DenseArrayType) -> DenseArrayType:
        """Evaluate image gradient at non-integer coordinates with linear interpolation

        Parameters
        ----------
        coordinates: (R, C, 2) array of float32
            The coordinates at which to interpolate. (R, C, 0) represent the y
            coordinates in the image, (R, C, 1) represent the x coordinate in
            the image

        Returns
        -------
        gradient: (R, C, 2) array of float32
            The interpolated gradients at all R, C coordinates. (R, C, 0) is
            the y coordinate of the gradient, (R, C, 1) is the x coordinate.
        """
        return self.evaluate_gradient_function(self.image, coordinates)


@njit(parallel=True)
def interpolate_cpu(
    image: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate image at non-integer coordinates with linear interpolation

    Parameters
    ----------
    image: (N, M) array of float32
        The array to use for interpolation
    coordinates: (R, C, 2) array of float32
        The coordinates at which to interpolate. (R, C, 0) represent the y
        coordinates in the image, (R, C, 1) represent the x coordinate in the
        image

    Returns
    -------
    values: (R, C) array of float32
        The interpolated values for all R, C coordinates
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


@njit(parallel=True)
def interpolate_gradient_cpu(
    image: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate image gradient at non-integer coordinates with linear interpolation

    Parameters
    ----------
    image: (N, M) array of float32
        The array to use for interpolation
    coordinates: (R, C, 2) array of float32
        The coordinates at which to interpolate. (R, C, 0) represent the y
        coordinates in the image, (R, C, 1) represent the x coordinate in the
        image

    Returns
    -------
    gradient: (R, C, 2) array of float32
        The interpolated gradients at all R, C coordinates. (R, C, 0) is the
        y coordinate of the gradient, (R, C, 1) is the x coordinate.
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
