import numpy as np
from math import prod
from typing import Optional, Tuple
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
    def cell_grid_shape(self) -> Tuple[int, int, int]:
        return self.quadrature.cell_grid_shape

    @property
    def number_of_quadrature_points(self) -> int:
        return self.quadrature.number_of_quadrature_points

    @cached_property
    def number_of_nodes(self) -> int:
        return prod(self.grid_shape)

    @cached_property
    def derivative_of_regularizer(self) -> SparseMatrixType:
        """Derivative of regularizer is constant matrix

        Has shape (4 * (N-1) * (M-1) * K, 2 * N * M), with the number of
        rows equaling 4x the total number of quadrature points and the
        columns 2x the total number of nodes (pixels).
        """
        # TODO: since the value going in is a constant, the number of unique
        # values is limited and there may be shortcuts to calculate this.
        dp = self.dispatcher
        sparse = self.sparse
        # regularization constant for each quadrature point in the grid of cells
        quadrature_values = dp.full(
            self.quadrature.cell_grid_shape,
            fill_value=self.regularization_constant_sqrt,
            dtype=dp.float32,
        )

        # reg = regularizer
        data_reg_x, rows_reg_x, cols_reg_x = (
            self.quadrature.evaluate_partial_derivatives(
                quadrature_values,
                node_weights=self.quadrature.basis_dfx_at_points,
            )
        )
        data_reg_y, rows_reg_y, cols_reg_y = (
            self.quadrature.evaluate_partial_derivatives(
                quadrature_values,
                node_weights=self.quadrature.basis_dfy_at_points,
            )
        )

        # combine the data into a single matrix
        n_quad_points = self.quadrature.total_number_of_quadrature_points
        block_shape = (2 * n_quad_points, self.image_reference.size)
        mat_reg = sparse.csr_matrix(
            (
                dp.concatenate((data_reg_x, data_reg_y)),
                (
                    dp.concatenate((rows_reg_x, rows_reg_y + n_quad_points)),
                    dp.concatenate((cols_reg_x, cols_reg_y)),
                ),
            ),
            shape=block_shape,
            dtype=dp.float32,
        )

        mat_zero = sparse.csr_matrix(block_shape, dtype=dp.float32)
        return sparse.vstack(
            [
                sparse.hstack([mat_zero, mat_reg]),
                sparse.hstack([mat_reg, mat_zero]),
            ]
        ).tocsr()

    def evaluate_residual(
        self,
        displacement_vector: DenseArrayType,
        positions_at_quad_points: Optional[DenseArrayType] = None,
    ) -> DenseArrayType:
        """Evaluate the error on the image corrected with the provided
        displacement with respect to the reference image

        Parameters
        ----------
        displacement_vector
            Array of length (2 * N * M), representing the y and x displacements
            in each pixel.
        positions_at_quad_points
            Array of shape ((N-1) * (M-1) * K, 2), representing the position
            field evaluated at all quadrature points. [:, 0] is the y component
            [:, 1] is the x component. It is computed from the displacement
            vector if not provided.

        Returns
        -------
        error
            Array of length (5 * (N-1) * (M-1) * K)
        """
        if positions_at_quad_points is None:
            positions_at_quad_points = self._quantize_displacement_vector(displacement_vector)

        dp = self.dispatcher
        R = self.cell_grid_shape[0]
        C = self.cell_grid_shape[1] * self.cell_grid_shape[2]
        positions_at_quad_points = positions_at_quad_points.reshape(R, C, 2)

        corrected_image = (
            self.image_deformed_interpolated
            .evaluate(positions_at_quad_points)
            .reshape(-1, self.number_of_quadrature_points)
        )

        ground_truth = (
            self.quadrature
            .evaluate(self.image_reference)
            .reshape(-1, self.number_of_quadrature_points)
        )

        residual_data = dp.multiply(
            self.quadrature.quadrature_point_weights_sqrt,
            corrected_image - ground_truth,
        )
        residual_regularization = self.derivative_of_regularizer.dot(displacement_vector)

        return dp.concatenate(
            (
                residual_data.ravel(),
                residual_regularization,
            )
        )

    def evaluate_residual_gradient(
        self,
        displacement_vector: DenseArrayType,
        positions_at_quad_points: Optional[DenseArrayType] = None,
    ) -> SparseMatrixType:
        """Evaluate the error on the corrected image with respect to the image_reference

        Parameters
        ----------
        displacement_vector
            Array of length (2 * N * M), representing the y and x displacements
            in each pixel.
        positions_at_quad_points
            Array of shape ((N-1) * (M-1) * K, 2), representing the position
            field evaluated at all quadrature points. [:, 0] is the y component
            [:, 1] is the x component. It is computed from the displacement
            vector if not provided.

        Returns
        -------
        error_gradient
            Sparse matrix of shape (5 * (N-1) * (M-1) * K, 2 * N * M)
        """
        if positions_at_quad_points is None:
            positions_at_quad_points = self._quantize_displacement_vector(displacement_vector)

        dp = self.dispatcher
        R = self.cell_grid_shape[0]
        C = self.cell_grid_shape[1] * self.cell_grid_shape[2]
        df = self.image_deformed_interpolated.evaluate_gradient(
            positions_at_quad_points.reshape(R, C, 2)
        ) / self.grid_scaling
        dfdy = df[..., 0].reshape(self.cell_grid_shape).astype(dp.float32)
        dfdx = df[..., 1].reshape(self.cell_grid_shape).astype(dp.float32)

        data_y, rows_y, cols_y = self.quadrature.evaluate_partial_derivatives(
            dfdy, self.quadrature.basis_f_at_points,
        )
        data_x, rows_x, cols_x = self.quadrature.evaluate_partial_derivatives(
            dfdx, self.quadrature.basis_f_at_points,
        )

        gradient_data = self.sparse.csr_matrix(
            (
                dp.concatenate((data_y, data_x)),
                (
                    dp.concatenate((rows_y, rows_x)),
                    dp.concatenate((cols_y, cols_x + self.number_of_nodes)),
                ),
            ),
            shape=(
                self.quadrature.total_number_of_quadrature_points,
                2 * self.number_of_nodes,
            ),
        )

        return self.sparse.vstack(
            [gradient_data, self.derivative_of_regularizer]
        ).tocsr()

    def evaluate_energy(
        self,
        displacement_vector: DenseArrayType,
        positions_at_quad_points: Optional[DenseArrayType] = None,
    ) -> float:
        dp = self.dispatcher
        return dp.sum(
            self.evaluate_residual(
                displacement_vector,
                positions_at_quad_points,
            ) ** 2
        )

    def evaluate_energy_gradient(
        self,
        displacement_vector: DenseArrayType,
        positions_at_quad_points: Optional[DenseArrayType] = None,
    ) -> DenseArrayType:
        residual = self.evaluate_residual(
            displacement_vector,
            positions_at_quad_points,
        )
        residual_gradient = self.evaluate_residual_gradient(
            displacement_vector,
            positions_at_quad_points,
        )
        return 2 * residual_gradient.T * residual.ravel()

    def _quantize_displacement_vector(
        self,
        displacement_vector: DenseArrayType,
    ) -> DenseArrayType:
        """Convert displacement field vector to quadrature point evaluations

        Parameters
        ----------
        displacement_vector
            Array of length (2 * N * M), representing the y and x displacements
            in each pixel.

        Returns
        -------
        positions_at_quad_points
            Array of shape ((N-1) * (M-1) * K, 2), representing the position
            field evaluated at all quadrature points. [:, 0] is the y component
            [:, 1] is the x component.
        """
        dp = self.dispatcher
        displacement_y, displacement_x = displacement_vector.reshape((2, *self.grid_shape))
        pixel_row, pixel_column = self.identity
        new_position_x = displacement_x / self.grid_scaling + pixel_column
        new_position_y = displacement_y / self.grid_scaling + pixel_row
        n_rows = self.quadrature.total_number_of_quadrature_points
        positions_at_quad_points = dp.empty((n_rows, 2), dtype=dp.float32)
        positions_at_quad_points[:, 0] = self.quadrature.evaluate(new_position_y).ravel()
        positions_at_quad_points[:, 1] = self.quadrature.evaluate(new_position_x).ravel()
        return positions_at_quad_points
