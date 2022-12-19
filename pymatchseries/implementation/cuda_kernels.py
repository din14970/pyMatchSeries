from typing import Tuple
import cupy as cp
from numba import cuda


def interpolate_gpu(
    image: cp.ndarray,
    coordinates: cp.ndarray,
) -> cp.ndarray:
    """Evaluate image at non-integer coordinates with linear interpolation
    """
    result = cp.empty(coordinates.shape[:2], dtype=cp.float32)
    bpg, tpb = _get_default_grid_dims_2D(coordinates)
    _evaluate_gpu_kernel[bpg, tpb](image, coordinates, result)
    return result


def interpolate_gradient_gpu(
    image: cp.ndarray,
    coordinates: cp.ndarray,
) -> cp.ndarray:
    """Evaluate image gradient at non-integer coordinates with linear interpolation
    """
    result = cp.zeros(coordinates.shape, dtype=cp.float32)
    bpg, tpb = _get_default_grid_dims_2D(coordinates)
    _evaluate_gradient_gpu_kernel[bpg, tpb](image, coordinates, result)
    return result


def _get_default_grid_dims_2D(
    array: cp.ndarray,
    tpb: Tuple[int, int] = (16, 16),
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Helper function for calculating grid dimensions for executing a CUDA kernel"""
    bpg = (
        (array.shape[0] + (tpb[0] - 1)) // tpb[0],
        (array.shape[1] + (tpb[1] - 1)) // tpb[1],
    )
    return bpg, tpb


@cuda.jit
def _evaluate_gpu_kernel(
    image: cp.ndarray,
    coordinates: cp.ndarray,
    result: cp.ndarray,
) -> None:
    """Evaluate image at non-integer coordinates with linear interpolation
    """
    row, column = cuda.grid(2)

    if row >= coordinates.shape[0] or column >= coordinates.shape[1]:
        return

    y = coordinates[row, column, 0]
    x = coordinates[row, column, 1]
    _, y0, wy = _get_interpolation_parameters(y, image.shape[0])
    _, x0, wx = _get_interpolation_parameters(x, image.shape[1])
    y1 = y0 + 1
    x1 = x0 + 1
    one_minus_wx = 1 - wx
    one_minus_wy = 1 - wy
    w_00 = one_minus_wx * one_minus_wy
    w_10 = wy * one_minus_wx
    w_01 = one_minus_wy * wx
    w_11 = wy * wx

    result[row, column] = (
        image[y0, x0] * w_00 +
        image[y1, x0] * w_10 +
        image[y0, x1] * w_01 +
        image[y1, x1] * w_11
    )


@cuda.jit
def _evaluate_gradient_gpu_kernel(
    image: cp.ndarray,
    coordinates: cp.ndarray,
    result: cp.ndarray,
) -> None:
    """Evaluate image gradient at non-integer coordinates with linear interpolation
    """
    row, column = cuda.grid(2)

    if row >= coordinates.shape[0] or column >= coordinates.shape[1]:
        return

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


@cuda.jit(device=True)
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
