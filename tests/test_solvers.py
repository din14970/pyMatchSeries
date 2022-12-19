import pytest
import numpy as np

from pymatchseries.utils import cp, CUPY_IS_INSTALLED
from pymatchseries.implementation import solvers


def test_sparse_cholesky():
    A = np.array(
        [[1., 2., 1.],
         [4., 5., 6.],
         [0., 8., 9.]],
    )
    M = np.dot(A.T, A)
    b = np.array([0.1, 0.5, -0.2])
    b = np.dot(A.T, b)
    x = solvers.solve_sparse_cholesky(M, b, np)
    assert np.allclose(M.dot(x), b)


@pytest.mark.skipif(not CUPY_IS_INSTALLED, reason="cupy not installed")
def test_sparse_cholesky_cp():
    A = cp.array(
        [[1., 2., 1.],
         [4., 5., 6.],
         [0., 8., 9.]],
    )
    M = cp.dot(A.T, A)
    b = cp.array([0.1, 0.5, -0.2])
    b = cp.dot(A.T, b)
    x = solvers.solve_sparse_cholesky(M, b, cp)
    assert cp.allclose(M.dot(x), b)


def test_sparse_cholesky_lq():
    A = np.array(
        [[1., 2., 3.],
         [4., 5., 6.],
         [7., 8., 9.],
         [7., 8., 9.]]
    )
    A_t = A.T
    b = np.array([0.1, 0.5, -0.2, -0.1])
    x = solvers.solve_sparse_cholesky_lq(A, b, np)
    MtM = np.dot(A_t, A)
    assert np.allclose(MtM.dot(x), A_t.dot(b))


@pytest.mark.skipif(not CUPY_IS_INSTALLED, reason="cupy not installed")
def test_sparse_cholesky_lq_cp():
    A = cp.array(
        [[1., 2., 3.],
         [4., 5., 6.],
         [7., 8., 9.],
         [7., 8., 9.]]
    )
    A_t = A.T
    b = cp.array([0.1, 0.5, -0.2, -0.1])
    x = solvers.solve_sparse_cholesky_lq(A, b, cp)
    MtM = cp.dot(A_t, A)
    assert cp.allclose(MtM.dot(x), A_t.dot(b))
