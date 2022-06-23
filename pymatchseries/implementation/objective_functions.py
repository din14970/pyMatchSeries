import numpy as np
from functools import cached_property
from scipy import sparse

from .interpolation import BilinearInterpolation2D
from .quadrature import Quadrature2D


class RegistrationObjectiveFunction:
    def __init__(
        self,
        image_deformed: np.ndarray,
        image_reference: np.ndarray,
        regularization_constant: float,
        number_of_quadrature_points: int = 3,
    ):
        if image_deformed.shape != image_reference.shape:
            raise ValueError(
                "Deformed and reference image are not the same shape: "
                "{image_deformed.shape} versus {image_reference.shape}."
            )
        self.grid_shape = (image_deformed.shape[0], image_deformed.shape[1])
        self.quadrature = Quadrature2D(
            number_of_points=number_of_quadrature_points,
            grid_shape=self.grid_shape,
        )

        self.image_deformed_interpolated = BilinearInterpolation2D(image_deformed)
        self.image_reference = image_reference

        self.regularization_constant_sqrt = np.sqrt(regularization_constant)

        self.identity = (
            np.mgrid[
                0: self.grid_shape[0],
                0: self.grid_shape[1]
            ].astype(np.float32)
        )

        self.regularization_constant = regularization_constant

    @cached_property
    def derivative_of_regularizer(self) -> sparse.spmatrix:
        quadeval = np.full(
            (
                self.grid_shape[0] - 1,
                self.grid_shape[1] - 1,
                self.quadrature.number_of_quadrature_points,
            ),
            fill_value=self.regularization_constant_sqrt,
            dtype=np.float32,
        )

        data_reg_x, rows_reg_x, cols_reg_x = _evaluate_pd_on_quad_points(
            quadeval, self.quad_weights_sqrt, self.dqvx
        )
        data_reg_y, rows_reg_y, cols_reg_y = _evaluate_pd_on_quad_points(
            quadeval, self.quad_weights_sqrt, self.dqvy
        )
        mat_reg = sparse.csr_matrix(
            (
                np.concatenate((data_reg_x, data_reg_y)),
                (
                    np.concatenate((rows_reg_x, rows_reg_y + quadeval.size)),
                    np.concatenate((cols_reg_x, cols_reg_y)),
                ),
            ),
            shape=(2 * quadeval.size, self.im2.size),
        )

        mat_zero = sparse.csr_matrix((2 * quadeval.size, self.im2.size))
        return sparse.vstack(
            [
                sparse.hstack([mat_zero, mat_reg]),
                sparse.hstack([mat_reg, mat_zero]),
            ]
        )

    def evaluate_residual(self, disp_vec):
        disp = disp_vec.reshape((2,) + self.grid_shape)
        return residual(
            disp[1, ...],
            disp[0, ...],
            self.im1_interp,
            self.im2,
            self.node_weights,
            self.quad_weights_sqrt,
            self.mat_reg_full,
            self.L_sqrt,
            self.identity,
            self.grid_h,
        )

    def evaluate_residual_gradient(self, disp_vec):
        disp = disp_vec.reshape((2,) + self.grid_shape)
        return residual_gradient(
            disp[1, ...],
            disp[0, ...],
            self.im1_interp,
            self.node_weights,
            self.quad_weights_sqrt,
            self.mat_reg_full,
            self.qv,
            self.identity,
            self.grid_h,
        )

    def evaluate_energy(self, disp_vec):
        # disp = disp_vec.reshape((2,) + self.grid_shape)
        # return energy(disp[1, ...], disp[0, ...], self.im1_interp, self.im2, self.node_weights, self.node_weights_dx, self.node_weights_dy, self.quad_weights, self.L, self.identity, self.grid_h)
        return np.sum(self.evaluate_residual(disp_vec) ** 2)

    def evaluate_energy_gradient(self, disp_vec):
        # disp = disp_vec.reshape((2,) + self.grid_shape)
        # return gradient(disp[1, ...], disp[0, ...], self.im1_interp, self.im2, self.node_weights, self.node_weights_dx, self.node_weights_dy, self.quad_weights, self.qv, self.dqvx, self.dqvy, self.L, self.identity, self.grid_h).ravel()
        res = self.evaluate_residual(disp_vec)
        mat = self.evaluate_residual_gradient(disp_vec)
        return 2 * mat.T * res.ravel()
