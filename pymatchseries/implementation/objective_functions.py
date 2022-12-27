import numpy as np
from typing import Sequence
from functools import cached_property
from scipy import sparse
from numba import njit

from scipy.sparse import csr_matrix, vstack

from .interpolation import BilinearInterpolation2D
from .quadrature import Quadrature2D

from pymatchseries.utils import DenseArrayType, get_dispatcher


class RegistrationObjectiveFunction:

    def __init__(
        self,
        image_deformed: DenseArrayType,
        image_reference: DenseArrayType,
        regularization_constant: float,
        number_of_quadrature_points: int = 3,
    ):
        self.grid_shape = image_deformed.shape
        self.dispatcher = get_dispatcher(image_deformed)
        self.quadrature = Quadrature2D(
            grid_shape=self.grid_shape,
            number_of_points=number_of_quadrature_points,
            dispatcher=self.dispatcher,
        )

        self.image_deformed_interpolated = BilinearInterpolation2D(image_deformed)
        self.image_reference = image_reference

        self.identity = self.dispatcher.mgrid[0: self.grid_shape[0], 0: self.grid_shape[1]].astype(np.float32)

        self.regularization_constant_sqrt = np.sqrt(regularization_constant)

    @cached_property
    def derivative_of_regularizer(self) -> sparse.spmatrix:
        dp = self.dispatcher
        quadeval = dp.full(
            (
                self.grid_shape[0] - 1,
                self.grid_shape[1] - 1,
                self.quadrature.number_of_quadrature_points,
            ),
            fill_value=self.regularization_constant_sqrt,
            dtype=dp.float32,
        )

        data_reg_x, rows_reg_x, cols_reg_x = _evaluate_pd_on_quad_points(
            quadeval, self.quad_weights_sqrt, self.dqvx
        )
        data_reg_y, rows_reg_y, cols_reg_y = _evaluate_pd_on_quad_points(
            quadeval, self.quad_weights_sqrt, self.dqvy
        )
        mat_reg = sparse.csr_matrix(
            (
                dp.concatenate((data_reg_x, data_reg_y)),
                (
                    dp.concatenate((rows_reg_x, rows_reg_y + quadeval.size)),
                    dp.concatenate((cols_reg_x, cols_reg_y)),
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
        pos_x = _value_at_quad_points(
            disp_x / self.grid_h + self.identity[1, ...], node_weights
        ).ravel()
        pos_y = _value_at_quad_points(
            disp_y / self.grid_h + self.identity[0, ...], node_weights
        ).ravel()
        g = _value_at_quad_points(im2, node_weights)
        # then we evaluate f(phi_x, phi_y)
        pos = np.stack((pos_y, pos_x), axis=-1)[np.newaxis, ...]
        f = im1_interp.evaluate(pos).reshape(-1, node_weights.shape[1])

        res_data = np.multiply(quad_weights_sqrt, (f - g))

        return np.concatenate(
            (
                res_data.ravel(),
                mat_reg_full * np.concatenate((disp_y.ravel(), disp_x.ravel())),
            )
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
        return np.sum(self.evaluate_residual(disp_vec) ** 2)

    def evaluate_energy_gradient(self, disp_vec):
        res = self.evaluate_residual(disp_vec)
        mat = self.evaluate_residual_gradient(disp_vec)
        return 2 * mat.T * res.ravel()


def residual_gradient(
    disp_x,
    disp_y,
    im1_interp,
    node_weights,
    quad_weights_sqrt,
    mat_reg_full,
    qv,
    identity,
    grid_h,
):
    pos_x = _value_at_quad_points(disp_x / grid_h + identity[1, ...], node_weights)
    pos_y = _value_at_quad_points(disp_y / grid_h + identity[0, ...], node_weights)
    pos = np.stack((pos_y, pos_x), axis=-1)
    cell_shape = (disp_x.shape[0] - 1, disp_x.shape[1] - 1, node_weights.shape[1])
    df = im1_interp.evaluate_gradient(pos) / grid_h
    dfdy = df[..., 0].reshape(cell_shape).astype(np.float32)
    dfdx = df[..., 1].reshape(cell_shape).astype(np.float32)
    data_y, rows_y, cols_y = _evaluate_pd_on_quad_points(dfdy, quad_weights_sqrt, qv)
    data_x, rows_x, cols_x = _evaluate_pd_on_quad_points(dfdx, quad_weights_sqrt, qv)

    mat_data = csr_matrix(
        (
            np.concatenate((data_y, data_x)),
            (
                np.concatenate((rows_y, rows_x)),
                np.concatenate((cols_y, cols_x + disp_x.size)),
            ),
        ),
        shape=(pos_x.size, 2 * disp_x.size),
    )

    return vstack([mat_data, mat_reg_full])


def residual(
    disp_x,
    disp_y,
    im1_interp,
    im2,
    node_weights,
    quad_weights_sqrt,
    mat_reg_full,
    L_sqrt,
    identity,
    grid_h,
) -> DenseArrayType:
    # we evaluate integral_over_domain (f(phi(x)) - g(x))**2 where x are all quad points (x_i, y_i)
    # first we evaluate (phi_x, phy_y) and g(x) at all quad points
    pos_x = _value_at_quad_points(
        disp_x / grid_h + identity[1, ...], node_weights
    ).ravel()
    pos_y = _value_at_quad_points(
        disp_y / grid_h + identity[0, ...], node_weights
    ).ravel()
    g = _value_at_quad_points(im2, node_weights)
    # then we evaluate f(phi_x, phi_y)
    pos = np.stack((pos_y, pos_x), axis=-1)[np.newaxis, ...]
    f = im1_interp.evaluate(pos).reshape(-1, node_weights.shape[1])

    res_data = np.multiply(quad_weights_sqrt, (f - g))

    return np.concatenate(
        (
            res_data.ravel(),
            mat_reg_full * np.concatenate((disp_y.ravel(), disp_x.ravel())),
        )
    )
