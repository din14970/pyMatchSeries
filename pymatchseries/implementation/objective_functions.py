import numpy as np
from functools import cached_property

from .interpolation import BilinearInterpolation2D
from .quadrature import Quadrature2D

from pymatchseries.utils import (
    DenseArrayType,
    SparseMatrixType,
    get_dispatcher,
    get_sparse_module,
)


class RegistrationObjectiveFunction:

    def __init__(
        self,
        image_deformed: DenseArrayType,
        image_reference: DenseArrayType,
        regularization_constant: float,
        number_of_quadrature_points: int = 3,
    ) -> None:
        self.dispatcher = get_dispatcher(image_deformed)
        self.grid_shape = image_deformed.shape
        self.sparse = get_sparse_module(self.dispatcher)
        self.quadrature = Quadrature2D(
            grid_shape=self.grid_shape,
            number_of_points=number_of_quadrature_points,
            dispatcher=self.dispatcher,
        )

        self.image_deformed_interpolated = BilinearInterpolation2D(image_deformed)
        self.image_reference = image_reference

        self.identity = self.dispatcher.mgrid[
            0: self.grid_shape[0],
            0: self.grid_shape[1],
        ].astype(np.float32)

        self.regularization_constant_sqrt = np.sqrt(regularization_constant)

    @property
    def grid_scaling(self) -> float:
        return self.quadrature.grid_scaling

    @property
    def number_of_quadrature_points(self) -> int:
        return self.quadrature.number_of_quadrature_points

    @cached_property
    def derivative_of_regularizer(self) -> SparseMatrixType:
        dp = self.dispatcher
        sparse = self.sparse
        # regularization constant for each quadrature point in the grid of cells
        quadrature_values = dp.full(
            (
                self.grid_shape[0] - 1,
                self.grid_shape[1] - 1,
                self.quadrature.number_of_quadrature_points,
            ),
            fill_value=self.regularization_constant_sqrt,
            dtype=dp.float32,
        )

        # reg = regularizer
        data_reg_x, rows_reg_x, cols_reg_x = (
            self.quadrature.evaluate_partial_derivatives(
                quadrature_values, self.quadrature.dx_node_weights
            )
        )
        data_reg_y, rows_reg_y, cols_reg_y = (
            self.quadrature.evaluate_partial_derivatives(
                quadrature_values, self.quadrature.dy_node_weights
            )
        )

        # combine the data into a single matrix
        mat_reg = sparse.csr_matrix(
            (
                dp.concatenate((data_reg_x, data_reg_y)),
                (
                    dp.concatenate((rows_reg_x, rows_reg_y + quadrature_values.size)),
                    dp.concatenate((cols_reg_x, cols_reg_y)),
                ),
            ),
            shape=(2 * quadrature_values.size, self.image_reference.size),
        )

        mat_zero = sparse.csr_matrix(
            (2 * quadrature_values.size, self.image_reference.size)
        )
        return sparse.vstack(
            [
                sparse.hstack([mat_zero, mat_reg]),
                sparse.hstack([mat_reg, mat_zero]),
            ]
        )

    def evaluate_residual(
        self,
        displacement_vector: DenseArrayType,
    ) -> DenseArrayType:
        """Evaluate the error on the corrected image with respect to the image_reference

        Parameters
        ----------
        displacement_vector

        Returns
        -------
        error
        """
        dp = self.dispatcher
        displacement_y, displacement_x = displacement_vector.reshape((2, *self.grid_shape))
        position_x = displacement_x / self.grid_scaling + self.identity[1, ...]
        position_y = displacement_y / self.grid_scaling + self.identity[0, ...]
        pos_x = self.quadrature.evaluate(position_x).ravel()
        pos_y = self.quadrature.evaluate(position_y).ravel()
        pos = dp.stack((pos_y, pos_x), axis=-1)[np.newaxis, ...]
        # then we evaluate f(phi_x, phi_y)
        corrected_image = (
            self.image_deformed_interpolated
            .evaluate(pos)
            .reshape(-1, self.number_of_quadrature_points)
        )

        ground_truth = self.quadrature.evaluate(self.image_reference)
        residual_data = dp.multiply(
            self.quadrature.quadrature_point_weights_sqrt,
            corrected_image - ground_truth,
        )
        residual_regularization = (
            self.derivative_of_regularizer *
            dp.concatenate((displacement_y.ravel(), displacement_x.ravel())),
        )

        return dp.concatenate(
            (
                residual_data.ravel(),
                residual_regularization,
            )
        )

    def evaluate_residual_gradient(
        self,
        displacement_vector: DenseArrayType,
    ) -> SparseMatrixType:
        """Evaluate the error on the corrected image with respect to the image_reference

        Parameters
        ----------
        displacement_vector:
        """
        dp = self.dispatcher
        displacement_y, displacement_x = displacement_vector.reshape((2, *self.grid_shape))
        position_x = displacement_x / self.grid_scaling + self.identity[1, ...]
        position_y = displacement_y / self.grid_scaling + self.identity[0, ...]
        pos_x = self.quadrature.evaluate(position_x)
        pos_y = self.quadrature.evaluate(position_y)
        pos = dp.stack((pos_y, pos_x), axis=-1)

        cell_grid_shape = (
            displacement_x.shape[0] - 1,
            displacement_y.shape[1] - 1,
            self.number_of_quadrature_points,
        )
        df = self.image_deformed_interpolated.evaluate_gradient(pos) / self.grid_scaling
        dfdy = df[..., 0].reshape(cell_grid_shape).astype(dp.float32)
        dfdx = df[..., 1].reshape(cell_grid_shape).astype(dp.float32)

        data_y, rows_y, cols_y = self.quadrature.evaluate_partial_derivatives(
            dfdy, self.quadrature.node_weights,
        )
        data_x, rows_x, cols_x = self.quadrature.evaluate_partial_derivatives(
            dfdx, self.quadrature.node_weights,
        )

        mat_data = self.sparse.csr_matrix(
            (
                dp.concatenate((data_y, data_x)),
                (
                    dp.concatenate((rows_y, rows_x)),
                    dp.concatenate((cols_y, cols_x + displacement_x.size)),
                ),
            ),
            shape=(pos_x.size, 2 * displacement_x.size),
        )

        return self.sparse.vstack([mat_data, self.derivative_of_regularizer])

    def evaluate_energy(
        self,
        displacement_vector: DenseArrayType,
    ) -> float:
        dp = self.dispatcher
        return dp.sum(self.evaluate_residual(displacement_vector) ** 2)

    def evaluate_energy_gradient(
        self,
        displacement_vector: DenseArrayType,
    ) -> DenseArrayType:
        res = self.evaluate_residual(displacement_vector)
        mat = self.evaluate_residual_gradient(displacement_vector)
        return 2 * mat.T * res.ravel()
