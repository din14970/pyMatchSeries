from typing import Callable, Mapping, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from pymatchseries.sorting import (
    heapsort_coo,
    heapsort_csr,
    insertion_sort_coo,
    insertion_sort_csr,
    introsort_coo,
    introsort_csr,
    quicksort_coo,
    quicksort_csr,
)

COO_DATA = Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]
SORTER_COO = Callable[[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]], None]
SORTER_CSR = Callable[[NDArray[np.int32], NDArray[np.float32]], None]


@pytest.fixture
def random_coo() -> COO_DATA:
    nnz = 150
    r = np.random.randint(0, 100, size=nnz, dtype=np.int32)
    c = np.random.randint(0, 150, size=nnz, dtype=np.int32)
    d = np.arange(nnz)
    return r, c, d


def _verify_result_coo_sort(
    r: NDArray[np.int32],
    c: NDArray[np.int32],
    d: NDArray[np.float32],
    data_pairs: Mapping[float, Tuple[int, int]],
) -> None:
    previous_r = 0
    previous_c = 0
    for row, column, data in zip(r, c, d):
        assert (row, column) >= (previous_r, previous_c)
        assert data_pairs[data] == (row, column)
        previous_r = row
        previous_c = column


@pytest.mark.parametrize(
    "algo",
    [
        heapsort_coo,
        insertion_sort_coo,
        introsort_coo,
        quicksort_coo,
    ],
)
def test_sort_coo(algo: SORTER_COO, random_coo: COO_DATA) -> None:
    r, c, d = random_coo
    data_pairs = {data: (row, column) for row, column, data in zip(r, c, d)}
    algo(r, c, d)
    _verify_result_coo_sort(r, c, d, data_pairs)


def _verify_result_csr_sort(
    c: NDArray[np.int32],
    d: NDArray[np.float32],
    data_pairs: Mapping[float, int],
) -> None:
    previous_c = 0
    for column, data in zip(c, d):
        assert column >= previous_c
        assert data_pairs[data] == column
        previous_c = column


@pytest.mark.parametrize(
    "algo",
    [
        heapsort_csr,
        insertion_sort_csr,
        introsort_csr,
        quicksort_csr,
    ],
)
def test_sort_csr(algo: SORTER_CSR, random_coo: COO_DATA) -> None:
    _, c, d = random_coo
    data_pairs = {data: column for column, data in zip(c, d)}
    algo(c, d)
    _verify_result_csr_sort(c, d, data_pairs)
