from typing import Tuple
from types import ModuleType

import numpy as np
from numba import njit, prange
from functools import cached_property

from pymatchseries.utils import DenseArrayType


class Quadrature2D:
    def __init__(
        self,
        grid_shape: Tuple[int, int],
        number_of_points: int = 3,
        dispatcher: ModuleType = np,
    ) -> None:
        """
        2D quadrature point representation to approximate the integral or
        gradient of a function F(x, y)
        """
        if number_of_points == 2:
            self._points = self._get_gauss_quad_points_2(dispatcher)
            self._weight = self._get_gauss_quad_weights_2(dispatcher)
        elif number_of_points == 3:
            self._points = self._get_gauss_quad_points_3(dispatcher)
            self._weight = self._get_gauss_quad_weights_3(dispatcher)
        else:
            raise NotImplementedError(
                f"Quadrature with {number_of_points} points not implemented",
            )
        self.grid_shape = grid_shape
        self.dispatcher = dispatcher
        self.grid_scaling: float = 1 / (max(grid_shape) - 1)
        if self.dispatcher == np:
            self.evaluate_function = evaluate_at_quad_points_cpu
            self.evaluate_pd_function = evaluate_pd_on_quad_points_cpu
        else:
            from pymatchseries.implementation.cuda_kernels import (
                evaluate_at_quad_points_gpu,
                evaluate_pd_on_quad_points_gpu,
            )
            self.evaluate_function = evaluate_at_quad_points_gpu
            self.evaluate_pd_function = evaluate_pd_on_quad_points_gpu

    def evaluate(self, array: DenseArrayType) -> DenseArrayType:
        """Get the value of an array interpolated at each quadrature point

        Parameters
        ----------
        array: (N, M) array of float32
            The array to evaluate the quad points on

        Returns
        -------
        values: ((N-1), (M-1), K) array of float32
            The interpolated value at each K quadrature point in each cell in
            the (N-1) x (M-1) grid
        """
        return self.evaluate_function(array, self.node_weights)

    def evaluate_partial_derivatives(
        self,
        quadrature_values: DenseArrayType,
        node_weights: DenseArrayType,
    ) -> Tuple[DenseArrayType, DenseArrayType, DenseArrayType]:
        """Get a sparse representation of the partial derivative at each quadrature points

        This represents a matrix of size (total number of quadrature points,
        total number of nodes in the original image). The number of nodes is
        equal to the number of basis functions.

        Parameters
        ----------
        quadrature_values: ((N-1), (M-1), K) array of float32
            The value of each K quadrature point in each of the (N-1) x (M-1) cells
        node_weights: (4, K) array of float32
            The weight each of the 4 surrounding nodes on each of the K quad points
            in a cell

        Returns
        -------
        data: ((N-1) x (M-1) x K x 4) array of float32
            Values in sparse array
        rows: ((N-1) x (M-1) x K x 4) array of int32
            Row indices in sparse matrix
        cols: ((N-1) x (M-1) x K x 4) array of int32
            Column indices in sparse matrix
        """
        return self.evaluate_pd_function(
            quadrature_values, self.quadrature_point_weights_sqrt, node_weights
        )

    @cached_property
    def quadrature_points(self) -> DenseArrayType:
        return self._points

    @cached_property
    def quadrature_point_weights(self) -> DenseArrayType:
        return self._weight * (self.grid_scaling ** 2)

    @cached_property
    def quadrature_point_weights_sqrt(self) -> DenseArrayType:
        return self.dispatcher.sqrt(self.quadrature_point_weights)

    @property
    def quadrature_points_x_coordinate(self) -> DenseArrayType:
        return self.quadrature_points[:, 0]

    @property
    def quadrature_points_y_coordinate(self) -> DenseArrayType:
        return self.quadrature_points[:, 1]

    @property
    def number_of_quadrature_points(self) -> int:
        return self.quadrature_points.shape[0]

    @cached_property
    def node_weights(self) -> DenseArrayType:
        """
        The weights w_i that each surrounding node contributes to evaluating
        the function f at the quadrature points at x, y, i.e.:
        `f(x,y) = w_0 * f_00 + w_1 * f_01 + w_2 * f_10 + w_4 * f_11.`
        """
        qx = self.quadrature_points_x_coordinate
        qy = self.quadrature_points_y_coordinate
        wx1 = 1 - qx
        wx2 = qx
        wy1 = 1 - qy
        wy2 = qy
        return self.dispatcher.vstack([
            wy1 * wx1,
            wy1 * wx2,
            wy2 * wx1,
            wy2 * wx2,
        ])

    @cached_property
    def dx_node_weights(self) -> DenseArrayType:
        """
        The weights to evaluate d/dx * f at x, y coordinates
        """
        qy = self.quadrature_points_y_coordinate
        one_minus_qy = 1 - qy
        return self.dispatcher.vstack(
            [-one_minus_qy, one_minus_qy, -qy, qy]
        ) / self.grid_scaling

    @cached_property
    def dy_node_weights(self) -> DenseArrayType:
        """
        The weights to evaluate d/dy * f at x, y coordinates
        """
        qx = self.quadrature_points_x_coordinate
        one_minus_qx = 1 - qx
        return self.dispatcher.vstack(
            [-one_minus_qx, -qx, one_minus_qx, qx]
        ) / self.grid_scaling

    @classmethod
    def _get_gauss_quad_points_2(
        cls,
        dispatcher: ModuleType = np,
    ) -> DenseArrayType:
        """
        Get the x, y coordinates of the Gaussian quadrature points with 4 points
        """
        p = 1 / dispatcher.sqrt(3) / 2
        quads = dispatcher.array(
            [
                [-p, -p],
                [p, -p],
                [-p, p],
                [p, p]
            ],
            dtype=dispatcher.float32,
        )
        quads += 0.5
        return quads

    @classmethod
    def _get_gauss_quad_weights_2(
        cls,
        dispatcher: ModuleType = np,
    ) -> DenseArrayType:
        """
        Get the weights for the Gaussian quadrature points with 4 points
        """
        return dispatcher.ones(4, dtype=dispatcher.float32) / 4

    @classmethod
    def _get_gauss_quad_points_3(
        cls,
        dispatcher: ModuleType = np,
    ) -> DenseArrayType:
        """
        Get the x, y coordinates of the Gaussian quadrature points with 9 points
        """
        p = dispatcher.sqrt(3 / 5) / 2
        quads = dispatcher.array(
            [
                [-p, -p],
                [0, -p],
                [p, -p],
                [-p, 0],
                [0, 0],
                [p, 0],
                [-p, p],
                [0, p],
                [p, p],
            ],
            dtype=dispatcher.float32,
        )
        quads += 0.5
        return quads

    @classmethod
    def _get_gauss_quad_weights_3(
        cls,
        dispatcher: ModuleType = np,
    ) -> DenseArrayType:
        """
        Get the weights for the Gaussian quadrature points with 9 points

        References
        ----------
        http://users.metu.edu.tr/csert/me582/ME582%20Ch%2003.pdf
        """
        return dispatcher.array(
            [25, 40, 25, 40, 64, 40, 25, 40, 25],
            dtype=dispatcher.float32,
        ) / 81 / 4


@njit(parallel=True)
def evaluate_at_quad_points_cpu(
    array: np.ndarray,
    node_weights: np.ndarray,
) -> np.ndarray:
    """Get the value of an array interpolated at each quadrature point

    Parameters
    ----------
    array: (N, M) array of float32
        The array to evaluate the quad points on
    node_weights: (4, K) of float32
        The weight each of the 4 surrounding nodes on each of the K quad points

    Returns
    -------
    values: ((N-1), (M-1), K) array of float32
        The value of each quadrature point in each of the cells
    """
    output = np.empty(
        (array.shape[0] - 1, array.shape[1] - 1, node_weights.shape[1]),
        dtype=np.float32,
    )
    for r in prange(array.shape[0] - 1):
        for c in range(array.shape[1] - 1):
            for p in range(node_weights.shape[1]):
                output[r, c, p] = (
                    array[r, c] * node_weights[0, p]
                    + array[r, c + 1] * node_weights[1, p]
                    + array[r + 1, c] * node_weights[2, p]
                    + array[r + 1, c + 1] * node_weights[3, p]
                )
    return output


@njit(parallel=True)
def evaluate_pd_on_quad_points_cpu(
    quadrature_values: np.ndarray,
    quad_weights_sqrt: np.ndarray,
    node_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # original data shape
    image_shape = (
        quadrature_values.shape[0] + 1,
        quadrature_values.shape[1] + 1,
    )

    number_of_values = 4 * quadrature_values.size
    data = np.empty(number_of_values, dtype=np.float32)
    rows = np.empty(number_of_values, dtype=np.int32)
    cols = np.empty(number_of_values, dtype=np.int32)
    col_offsets = np.array(
        [image_shape[1] + 1, image_shape[1], 1, 0],
        dtype=np.int32,
    )
    for i in prange(quadrature_values.shape[0]):
        for j in prange(quadrature_values.shape[1]):
            # index in flattened 2D array
            abs_2D = j + i * quadrature_values.shape[1]
            col_base_index = abs_2D + i
            offset_2D = quadrature_values.shape[2] * abs_2D
            for k in range(quadrature_values.shape[2]):
                # index in flattened 3D array
                abs_3D = k + offset_2D
                offset_3D = 4 * abs_3D
                val = quadrature_values[i, j, k] * quad_weights_sqrt[k]
                for node in range(4):
                    idx = node + offset_3D
                    data[idx] = val * node_weights[node, k]
                    rows[idx] = abs_3D
                    cols[idx] = col_base_index + col_offsets[node]

    return data, rows, cols
