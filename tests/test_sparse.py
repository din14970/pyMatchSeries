from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

import pymatchseries.sparse as psparse


def _random_sparse_rand(
    shape: Tuple[int, int],
    fill_fraction: float,
) -> Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    size = shape[0] * shape[1]
    nn = int(fill_fraction * size)
    indexes = np.random.choice(size, replace=False, size=(nn,))
    rows = indexes // shape[1]
    columns = indexes % shape[1]
    data = np.random.rand(nn).astype(np.float32)
    return rows.astype(np.int32), columns.astype(np.int32), data


def _create_coo(
    shape: Tuple[int, int],
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> sparse.coo_matrix:
    return sparse.coo_matrix((data, (rows, columns)), shape=shape)


def test_csr_to_csc() -> None:
    shape = (100, 110)
    r, c, d = _random_sparse_rand(shape, 0.1)
    coo = _create_coo(shape, r, c, d)
    A_csr = coo.tocsr()
    A_csc = coo.tocsc()
    r_csc, c_csc, d_csc = psparse.csr_to_csc(
        shape[0],
        shape[1],
        A_csr.indptr,
        A_csr.indices,
        A_csr.data,
    )
    np.testing.assert_equal(r_csc, A_csc.indptr)
    np.testing.assert_equal(c_csc, A_csc.indices)
    np.testing.assert_equal(d_csc, A_csc.data)


def test_csr_sort() -> None:
    r = np.array([0, 3, 5, 10])
    c = np.array([3, 2, 1, 5, 4, 10, 9, 8, 7, 6])
    d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    c_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    d_test = np.array([2, 1, 0, 4, 3, 9, 8, 7, 6, 5])

    psparse.csr_sort_indices(r, c, d)

    np.testing.assert_equal(c, c_test)
    np.testing.assert_equal(d, d_test)


def test_coo_to_csr() -> None:
    shape = (100, 110)
    r, c, d = _random_sparse_rand(shape, 0.1)

    coo = _create_coo(shape, r, c, d)
    A_csr = coo.tocsr()

    r_csr, c_csr, d_csr = psparse.coo_to_csr(
        shape[0],
        rows=r,
        columns=c,
        data=d,
    )
    psparse.csr_sort_indices(r_csr, c_csr, d_csr)

    np.testing.assert_equal(r_csr, A_csr.indptr)
    np.testing.assert_equal(c_csr, A_csr.indices)
    np.testing.assert_equal(d_csr, A_csr.data)
