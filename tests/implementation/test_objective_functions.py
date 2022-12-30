import pytest
import numpy as np
from pymatchseries.implementation.objective_functions import (
    RegistrationObjectiveFunction,
)
from pymatchseries.utils import CUPY_IS_INSTALLED, cp
from scipy import sparse

from pathlib import Path

this_file = Path(__file__)
data_folder = this_file.parent.parent / "data"


image_deformed = np.array(
    [
        [9, 8, 0, 5, 5, 5],
        [4, 7, 9, 5, 2, 3],
        [9, 4, 5, 7, 1, 8],
        [8, 1, 2, 7, 2, 6],
        [8, 2, 9, 7, 0, 3],
    ]
).astype(np.float32)

image_reference = np.array(
    [
        [3, 4, 2, 6, 9, 0],
        [5, 6, 2, 4, 7, 1],
        [0, 5, 1, 2, 7, 2],
        [0, 7, 2, 0, 0, 8],
        [6, 4, 0, 9, 7, 8],
    ]
).astype(np.float32)
regularization_constant = 10
number_of_quadrature_points = 3

regobj_cpu = RegistrationObjectiveFunction(
    image_deformed=image_deformed,
    image_reference=image_reference,
    regularization_constant=regularization_constant,
    number_of_quadrature_points=number_of_quadrature_points,
)

params = [regobj_cpu]

if CUPY_IS_INSTALLED:
    image_deformed_gpu = cp.asarray(image_deformed)
    image_reference_gpu = cp.asarray(image_reference)
    regobj_gpu = RegistrationObjectiveFunction(
        image_deformed=image_deformed_gpu,
        image_reference=image_reference_gpu,
        regularization_constant=regularization_constant,
        number_of_quadrature_points=number_of_quadrature_points,
    )
    params.append(regobj_gpu)


DATA = {
    "expected_derivative": sparse.load_npz(
        data_folder / "expected_derivative_regularizer.npz"
    ),
    "expected_eval_residual": np.load(data_folder / "expected_eval_residual.npy"),
    "expected_eval_residual_grad": sparse.load_npz(
        data_folder / "expected_eval_residual_grad.npz"
    ),
    "expected_energy_gradient": np.load(data_folder / "expected_energy_gradient.npy"),
}

RTOL = 1e-6
ATOL = 1e-6


@pytest.mark.parametrize("of", params)
class TestRegistrationObjectiveFunction:
    def test_derivative_of_regularizer(
        self,
        of: RegistrationObjectiveFunction,
    ) -> None:
        dreg = of.derivative_of_regularizer
        expected_shape = (
            4 * of.quadrature.total_number_of_quadrature_points,
            2 * of.number_of_nodes,
        )
        assert dreg.shape == expected_shape

        if of.dispatcher != np:
            dreg = dreg.get()

        np.testing.assert_allclose(
            DATA["expected_derivative"].data, dreg.data, rtol=RTOL
        )
        np.testing.assert_array_equal(DATA["expected_derivative"].indices, dreg.indices)
        np.testing.assert_array_equal(DATA["expected_derivative"].indptr, dreg.indptr)

    def test_evaluate_residual(
        self,
        of: RegistrationObjectiveFunction,
    ) -> None:
        dp = of.dispatcher
        v = self.displacement_vector
        if dp != np:
            v = cp.asarray(v)
        error = of.evaluate_residual(v)
        expected_shape = (5 * of.quadrature.total_number_of_quadrature_points,)
        assert error.shape == expected_shape

        if dp != np:
            error = error.get()

        np.testing.assert_allclose(
            DATA["expected_eval_residual"], error, rtol=RTOL, atol=ATOL
        )

    def test_evaluate_residual_gradient(
        self,
        of: RegistrationObjectiveFunction,
    ) -> None:
        dp = of.dispatcher
        v = self.displacement_vector
        if dp != np:
            v = cp.asarray(v)
        error_grad = of.evaluate_residual_gradient(v)
        expected_shape = (
            5 * of.quadrature.total_number_of_quadrature_points,
            2 * of.number_of_nodes,
        )
        assert error_grad.shape == expected_shape

        if dp != np:
            error_grad = error_grad.get()

        np.testing.assert_allclose(
            DATA["expected_eval_residual_grad"].data,
            error_grad.data,
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_array_equal(
            DATA["expected_eval_residual_grad"].indices, error_grad.indices
        )
        np.testing.assert_array_equal(
            DATA["expected_eval_residual_grad"].indptr, error_grad.indptr
        )

    def test_evaluate_energy(
        self,
        of: RegistrationObjectiveFunction,
    ) -> None:
        dp = of.dispatcher
        v = self.displacement_vector
        if dp != np:
            v = cp.asarray(v)
        error = of.evaluate_energy(v)

        if dp != np:
            error = error.get()

        expected = 97.65085868419155

        assert abs(error - expected) / expected < RTOL

    def test_evaluate_energy_gradient(
        self,
        of: RegistrationObjectiveFunction,
    ) -> None:
        dp = of.dispatcher
        v = self.displacement_vector
        if dp != np:
            v = cp.asarray(v)
        error_grad = of.evaluate_energy_gradient(v)
        expected_shape = (2 * of.number_of_nodes,)
        assert error_grad.shape == expected_shape

        if dp != np:
            error_grad = error_grad.get()

        np.testing.assert_allclose(
            DATA["expected_energy_gradient"],
            error_grad,
            rtol=RTOL,
            atol=ATOL,
        )

    # Some data

    # random displacement vector
    displacement_vector = np.array(
        [
            0.2,
            0.4,
            0.7,
            0.5,
            0.7,
            0.7,
            0.3,
            0.1,
            0.1,
            0.9,
            0.7,
            0.8,
            0.7,
            0.3,
            0.7,
            0.1,
            0.5,
            0.8,
            0.2,
            0.9,
            0.3,
            0.9,
            0.5,
            0.3,
            0.2,
            0.6,
            0.6,
            0.1,
            0.6,
            0.5,
            0.2,
            0.1,
            0.7,
            0.0,
            0.6,
            0.4,
            0.4,
            0.3,
            0.8,
            0.0,
            0.2,
            0.7,
            0.6,
            0.6,
            0.4,
            0.9,
            0.8,
            0.8,
            0.5,
            0.1,
            0.1,
            0.4,
            0.0,
            0.6,
            0.2,
            0.4,
            0.3,
            0.6,
            0.9,
            0.1,
        ]
    ).astype(np.float32)
