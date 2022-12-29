from typing import Tuple
import cupy as cp
from numba import cuda, float32


TPB = 16
TPB1 = TPB + 1


def interpolate_gpu(
    image: cp.ndarray,
    coordinates: cp.ndarray,
) -> cp.ndarray:
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
    result = cp.empty(coordinates.shape[:2], dtype=cp.float32)
    bpg, tpb = _get_default_grid_dims_2D(result)
    _evaluate_gpu_kernel[bpg, tpb](image, coordinates, result)
    return result


def interpolate_gradient_gpu(
    image: cp.ndarray,
    coordinates: cp.ndarray,
) -> cp.ndarray:
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
    result = cp.zeros(coordinates.shape, dtype=cp.float32)
    bpg, tpb = _get_default_grid_dims_2D(result, tpb=(TPB, TPB))
    _evaluate_gradient_gpu_kernel[bpg, tpb](image, coordinates, result)
    return result


def evaluate_at_quad_points_gpu(
    array: cp.ndarray,
    node_weights: cp.ndarray,
) -> cp.ndarray:
    """Get the value of an array interpolated at each quadrature point

    Parameters
    ----------
    array: (N, M) array
        The array to evaluate the quad points on
    node_weights: (4, K) of float32
        The weight each of the 4 surrounding nodes on each of the K quad points

    Returns
    -------
    values: ((N-1), (M-1), K) array of float32
        The value of each quadrature point in each of the cells
    """
    output = cp.empty(
        (array.shape[0] - 1, array.shape[1] - 1, node_weights.shape[1]),
        dtype=cp.float32,
    )
    bpg, tpb = _get_default_grid_dims_2D(output)
    _evaluate_at_quad_points_kernel[bpg, tpb](array, node_weights, output)
    return output


def evaluate_pd_on_quad_points_gpu(
    quadrature_values: cp.ndarray,
    quad_weights_sqrt: cp.ndarray,
    node_weights: cp.ndarray,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Get a sparse representation of each node contribution to each quadrature point

    This represents a matrix of size (total number of quadrature points,
    total number of nodes in the original image)

    Parameters
    ----------
    quadrature_values: ((N-1), (M-1), K) array of float32
        The value of each K quadrature point in each of the (N-1) x (M-1) cells
    quad_weights_sqrt: (K,) array of float32
        Square root of the weight of each quadrature point
    node_weights: (4, K) array of float32
        The weight each of the 4 surrounding nodes on each of the K quad points

    Returns
    -------
    data: ((N-1) x (M-1) x K x 4) array of float32
        Values in sparse array
    rows: ((N-1) x (M-1) x K x 4) array of int32
        Row indices in sparse matrix
    cols: ((N-1) x (M-1) x K x 4) array of int32
        Column indices in sparse matrix
    """
    number_of_values = 4 * quadrature_values.size
    data = cp.empty(number_of_values, dtype=cp.float32)
    rows = cp.empty(number_of_values, dtype=cp.int32)
    cols = cp.empty(number_of_values, dtype=cp.int32)
    bpg, tpb = _get_default_grid_dims_2D(quadrature_values)
    _evaluate_pd_on_quad_points_kernel[bpg, tpb](
        quadrature_values,
        quad_weights_sqrt,
        node_weights,
        data,
        rows,
        cols,
    )
    return data, rows, cols


def _get_default_grid_dims_2D(
    array: cp.ndarray,
    tpb: Tuple[int, int] = (TPB, TPB),
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
    """Determine if a coordinate is within bounds, and what its weight is"""
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


@cuda.jit
def _evaluate_at_quad_points_kernel(
    array: cp.ndarray,
    node_weights: cp.ndarray,
    output: cp.ndarray
) -> None:
    r, c = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # all threads require the same node weights, and often same image pixels
    # TODO: also put node weights into a shared array
    s_array = cuda.shared.array(shape=(TPB1, TPB1), dtype=float32)

    if r >= output.shape[0] or c >= output.shape[1]:
        return

    s_array[tx, ty] = array[r, c]
    TPBM = TPB - 1
    at_bottom_of_block = (tx == TPBM or r == output.shape[0] - 1)
    at_right_of_block = (ty == TPBM or c == output.shape[1] - 1)

    if at_bottom_of_block:
        s_array[tx + 1, ty] = array[r + 1, c]
    if at_right_of_block:
        s_array[tx, ty + 1] = array[r, c + 1]
    if at_right_of_block and at_bottom_of_block:
        s_array[tx + 1, ty + 1] = array[r + 1, c + 1]

    cuda.syncthreads()

    for p in range(node_weights.shape[1]):
        output[r, c, p] = (
            s_array[tx, ty] * node_weights[0, p]
            + s_array[tx, ty + 1] * node_weights[1, p]
            + s_array[tx + 1, ty] * node_weights[2, p]
            + s_array[tx + 1, ty + 1] * node_weights[3, p]
        )


@cuda.jit
def _evaluate_pd_on_quad_points_kernel(
    quadrature_values: cp.ndarray,
    quad_weights_sqrt: cp.ndarray,
    node_weights: cp.ndarray,
    data: cp.ndarray,
    rows: cp.ndarray,
    cols: cp.ndarray,
) -> None:
    i, j = cuda.grid(2)

    if i >= quadrature_values.shape[0] or j >= quadrature_values.shape[1]:
        return

    # original data shape
    image_shape = (
        quadrature_values.shape[0] + 1,
        quadrature_values.shape[1] + 1,
    )
    col_offsets = (image_shape[1] + 1, image_shape[1], 1, 0)

    abs_2D = j + i * quadrature_values.shape[1]
    col_base_index = abs_2D + i
    offset_2D = quadrature_values.shape[2] * abs_2D
    for k in range(quadrature_values.shape[2]):
        abs_3D = k + offset_2D
        offset_3D = 4 * abs_3D
        val = quadrature_values[i, j, k] * quad_weights_sqrt[k]
        for node in range(4):
            idx = node + offset_3D
            data[idx] = val * node_weights[node, k]
            rows[idx] = abs_3D
            cols[idx] = col_base_index + col_offsets[node]
