import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def introsort_csr(
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> None:
    """Sort data for column coordinates in a CSR matrix inplace"""
    maxdepth = int(np.log2(data.shape[0])) * 2
    _introsort_csr(columns, data, maxdepth)


@njit
def _introsort_csr(
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    maxdepth: int,
) -> None:
    n = data.shape[0]
    if n < 16:
        insertion_sort_csr(columns, data)
    elif maxdepth == 0:
        heapsort_csr(columns, data)
    else:
        p = _partition_csr(columns, data, 0, n - 1)
        _introsort_csr(columns[:p], data[:p], maxdepth - 1)
        _introsort_csr(columns[p + 1 :], data[p + 1 :], maxdepth - 1)


@njit
def insertion_sort_csr(
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> None:
    n = data.shape[0]
    i = 1
    while i < n:
        j = i
        while (j > 0) and (columns[j] < columns[j - 1]):
            _swap(columns, j - 1, j)
            _swap(data, j - 1, j)
            j -= 1
        i = i + 1


@njit
def _partition_csr(
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    lo: int,
    hi: int,
) -> int:
    i = lo - 1

    for j in range(lo, hi):
        if columns[j] <= columns[hi]:
            i = i + 1
            _swap(columns, i, j)
            _swap(data, i, j)

    i += 1
    _swap(columns, i, hi)
    _swap(data, i, hi)
    return i


@njit
def quicksort_csr(
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> None:
    _quicksort_csr(columns, data, 0, data.shape[0] - 1)


@njit
def _quicksort_csr(
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    lo: int,
    hi: int,
) -> None:
    if (lo >= hi) or (lo < 0):
        return

    p = _partition_csr(columns, data, lo, hi)

    _quicksort_csr(columns, data, lo, p - 1)
    _quicksort_csr(columns, data, p + 1, hi)


@njit
def heapsort_csr(
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> None:
    count = data.shape[0]
    _heapify_csr(columns, data, count)
    end = count - 1
    while end > 0:
        _swap(columns, 0, end)
        _swap(data, 0, end)

        end -= 1
        _siftdown_csr(columns, data, 0, end)


@njit
def _heapify_csr(
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    count: int,
) -> None:
    start = int((count - 2) / 2)

    while start >= 0:
        _siftdown_csr(columns, data, start, count - 1)
        start -= 1


@njit
def _siftdown_csr(
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    start: int,
    end: int,
) -> None:
    root = start

    while 2 * root + 1 <= end:
        child = 2 * root + 1
        swap_i = root

        if columns[swap_i] < columns[child]:
            swap_i = child
        if (child + 1 <= end) and (columns[swap_i] < columns[child + 1]):
            swap_i = child + 1
        if swap_i == root:
            return
        else:
            _swap(columns, root, swap_i)
            _swap(data, root, swap_i)
            root = swap_i


@njit
def introsort_coo(
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> None:
    """Sort the COO triples according to (row, column) pairs using introsort.
    In-place operation.
    """
    maxdepth = int(np.log2(data.shape[0])) * 2
    _introsort_coo(rows, columns, data, maxdepth)


@njit
def _introsort_coo(
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    maxdepth: int,
) -> None:
    n = data.shape[0]
    if n < 16:
        insertion_sort_coo(rows, columns, data)
    elif maxdepth == 0:
        heapsort_coo(rows, columns, data)
    else:
        p = _partition_coo(rows, columns, data, 0, n - 1)
        _introsort_coo(rows[:p], columns[:p], data[:p], maxdepth - 1)
        _introsort_coo(rows[p + 1 :], columns[p + 1 :], data[p + 1 :], maxdepth - 1)


@njit
def insertion_sort_coo(
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> None:
    n = rows.shape[0]
    i = 1
    while i < n:
        j = i
        while (j > 0) and ((rows[j], columns[j]) < (rows[j - 1], columns[j - 1])):
            _swap(rows, j - 1, j)
            _swap(columns, j - 1, j)
            _swap(data, j - 1, j)
            j -= 1
        i = i + 1


@njit
def _partition_coo(
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    lo: int,
    hi: int,
) -> int:
    i = lo - 1

    for j in range(lo, hi):
        if (rows[j], columns[j]) <= (rows[hi], columns[hi]):
            i = i + 1
            _swap(rows, i, j)
            _swap(columns, i, j)
            _swap(data, i, j)

    i += 1
    _swap(rows, i, hi)
    _swap(columns, i, hi)
    _swap(data, i, hi)
    return i


@njit
def quicksort_coo(
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> None:
    _quicksort_coo(rows, columns, data, 0, data.shape[0] - 1)


@njit
def _quicksort_coo(
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    lo: int,
    hi: int,
) -> None:
    if (lo >= hi) or (lo < 0):
        return

    p = _partition_coo(rows, columns, data, lo, hi)

    _quicksort_coo(rows, columns, data, lo, p - 1)
    _quicksort_coo(rows, columns, data, p + 1, hi)


@njit
def _swap(A: NDArray, i: int, j: int) -> None:
    A[i], A[j] = A[j], A[i]


@njit
def heapsort_coo(
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> None:
    count = data.shape[0]
    _heapify_coo(rows, columns, data, count)
    end = count - 1
    while end > 0:
        _swap(rows, 0, end)
        _swap(columns, 0, end)
        _swap(data, 0, end)

        end -= 1
        _siftdown_coo(rows, columns, data, 0, end)


@njit
def _heapify_coo(
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    count: int,
) -> None:
    start = int((count - 2) / 2)

    while start >= 0:
        _siftdown_coo(rows, columns, data, start, count - 1)
        start -= 1


@njit
def _siftdown_coo(
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    start: int,
    end: int,
) -> None:
    root = start

    while 2 * root + 1 <= end:
        child = 2 * root + 1
        swap_i = root

        if (rows[swap_i], columns[swap_i]) < (rows[child], columns[child]):
            swap_i = child
        if (child + 1 <= end) and (rows[swap_i], columns[swap_i]) < (
            rows[child + 1],
            columns[child + 1],
        ):
            swap_i = child + 1
        if swap_i == root:
            return
        else:
            _swap(rows, root, swap_i)
            _swap(columns, root, swap_i)
            _swap(data, root, swap_i)
            root = swap_i
