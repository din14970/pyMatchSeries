from typing import Tuple

import numpy as np
from functools import cached_property


class Quadrature2D:
    def __init__(
        self,
        number_of_points: int = 3,
        grid_shape: Tuple[int, int] = (2, 2),
    ):
        """
        2D quadrature point representation to approximate the integral or
        gradient of a function F(x, y)
        """
        if number_of_points == 2:
            self._points = self._get_gauss_quad_points_2()
            self._weight = self._get_gauss_quad_weights_2()
        elif number_of_points == 3:
            self._points = self._get_gauss_quad_points_3()
            self._weight = self._get_gauss_quad_weights_3()
        else:
            raise NotImplementedError(f"Quadrature with {number_of_points} not implemented")
        self.grid_shape = grid_shape
        self.grid_scaling: float = 1 / (max(grid_shape) - 1)

    @cached_property
    def quadrature_points(self) -> np.ndarray:
        return self._points

    @cached_property
    def quadrature_point_weights(self) -> np.ndarray:
        return self._weight * (self.grid_scaling ** 2)

    @cached_property
    def quadrature_point_weights_sqrt(self) -> np.ndarray:
        return np.sqrt(self.quadrature_point_weights)

    @cached_property
    def quadrature_points_x_coordinate(self) -> np.ndarray:
        return self.quadrature_points[:, 0]

    @cached_property
    def quadrature_points_y_coordinate(self) -> np.ndarray:
        return self.quadrature_points[:, 1]

    @property
    def number_of_quadrature_points(self) -> int:
        return self.quadrature_points.shape[0]

    @cached_property
    def basis_function_value_at_quadrature_points(self) -> np.ndarray:
        """
        The value of the four basis functions at adjacent cells of a node
        evaluated at all of the quadrature points used throughout to evaluate
        functions and derivatives at the quad points.
        """
        qv = np.empty((4, self.number_of_quadrature_points), dtype=np.float32)
        qx = self.quadrature_points_x_coordinate
        qy = self.quadrature_points_y_coordinate
        qv[0, :] = qx * qy  # top left of node
        qv[1, :] = (1 - qx) * qy  # top right of node
        qv[2, :] = (1 - qy) * qx  # bottom left of node
        qv[3, :] = (1 - qx) * (1 - qy)  # bottom right of node
        return qv

    @cached_property
    def basis_function_gradient_at_quadrature_points(self) -> np.ndarray:
        """
        The gradient of the four basis function at adjacent cells of a node
        evaluated at all of the quadrature points
        """
        dqv = np.empty((4, 2, self.number_of_quadrature_points), dtype=np.float32)
        qx = self.quadrature_points_x_coordinate
        qy = self.quadrature_points_y_coordinate
        dqv[0, :, :] = (qy, qx)  # top left of node
        dqv[1, :, :] = (-qy, (1 - qx))  # top right of node
        dqv[2, :, :] = ((1 - qy), -qx)  # bottom left of node
        dqv[3, :, :] = (-(1 - qy), -(1 - qx))  # bottom right of node
        return dqv / self.grid_scaling

    @cached_property
    def basis_function_dx_at_quadrature_points(self) -> np.ndarray:
        return self.basis_function_gradient_at_quadrature_points[:, 0, :]

    @cached_property
    def basis_function_dy_at_quadrature_points(self) -> np.ndarray:
        return self.basis_function_gradient_at_quadrature_points[:, 1, :]

    @cached_property
    def node_weights(self) -> np.ndarray:
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
        return np.vstack([wy1 * wx1, wy1 * wx2, wy2 * wx1, wy2 * wx2])

    @cached_property
    def dx_node_weights(self) -> np.ndarray:
        """
        The weights to evaluate d/dx * f at x, y coordinates
        """
        qy = self.quadrature_points_y_coordinate
        return np.vstack([-(1 - qy), 1 - qy, -qy, qy]) / self.grid_scaling

    @cached_property
    def dy_node_weights(self) -> np.ndarray:
        """
        The weights to evaluate d/dy * f at x, y coordinates
        """
        qx = self.quadrature_points_x_coordinate
        return np.vstack([-(1 - qx), -qx, 1 - qx, qx]) / self.grid_scaling

    @classmethod
    def _get_gauss_quad_points_2(cls) -> np.ndarray:
        """
        Get the x, y coordinates of the Gaussian quadrature points with 4 points
        """
        p = 1 / np.sqrt(3) / 2
        quads = np.array([[-p, -p], [p, -p], [-p, p], [p, p]], dtype=np.float32)
        quads += 0.5
        return quads

    @classmethod
    def _get_gauss_quad_weights_2(cls) -> np.ndarray:
        """
        Get the weights for the Gaussian quadrature points with 4 points
        """
        return np.ones(4, dtype=np.float32) / 4

    @classmethod
    def _get_gauss_quad_points_3(cls) -> np.ndarray:
        """
        Get the x, y coordinates of the Gaussian quadrature points with 9 points
        """
        p = np.sqrt(3 / 5) / 2
        quads = np.array(
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
            dtype=np.float32,
        )
        quads += 0.5
        return quads

    @classmethod
    def _get_gauss_quad_weights_3(cls) -> np.ndarray:
        """
        Get the weights for the Gaussian quadrature points with 9 points

        References
        ----------
        http://users.metu.edu.tr/csert/me582/ME582%20Ch%2003.pdf
        """
        return np.array([25, 40, 25, 40, 64, 40, 25, 40, 25], dtype=np.float32) / 81 / 4
