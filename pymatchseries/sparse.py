from typing import Tuple

import numpy as np
from numba import njit, float32, int32
from numpy.typing import NDArray

from pymatchseries.sorting import introsort_csr

COOMatrix = Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]
CSRMatrix = Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]
CSCMatrix = Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]


def csr_sort_indices(
    row_ptr: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> None:
    n_rows = row_ptr.shape[0] - 1

    for row in range(n_rows):
        row_start = row_ptr[row]
        row_end = row_ptr[row + 1]

        introsort_csr(
            columns[row_start: row_end],
            data[row_start: row_end],
        )


@njit
def csc_to_csr(
    n_row: int,
    n_col: int,
    col_ptr: NDArray[np.int32],
    rows: NDArray[np.int32],
    data: NDArray[np.float32],
) -> CSRMatrix:
    return csr_to_csc(
        n_row=n_col,
        n_col=n_row,
        row_ptr=col_ptr,
        columns=rows,
        data=data,
    )


@njit
def csr_to_csc(
    n_row: int,
    n_col: int,
    row_ptr: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> CSCMatrix:
    """Calculate the CSC representation from a CSR representation

    Parameters
    ----------
    n_row
        The number of rows in the matrix
    n_col
        The number of columns in the matrix
    row_ptr
        The index pointer array of the CSR matrix
    columns
        The column values of the CSR matrix
    data
        The data values in the CSR matrix

    Returns
    -------
    col_ptr_out
        The index pointer array of the CSC matrix
    rows_out
        The row values of the CSC matrix
    data_out
        The data values in the CSC matrix

    Notes
    -----
    Numba implementation of the C++ version found in the Scipy source code
    """
    col_ptr_out = row_ptr_from_coo(n_col, columns)
    rows_out = np.empty_like(columns)
    data_out = np.empty_like(data)

    for row in range(n_row):
        for nze_index in range(row_ptr[row], row_ptr[row + 1]):
            col = columns[nze_index]
            nze_index_out = col_ptr_out[col]

            rows_out[nze_index_out] = row
            data_out[nze_index_out] = data[nze_index]

            # shift the pointer for that row by 1
            col_ptr_out[col] += 1

    # shift all pointers back
    col_ptr_out[1:] = col_ptr_out[:n_col]
    col_ptr_out[0] = 0

    return col_ptr_out, rows_out, data_out


@njit
def csr_matmat(
    n_row: int,
    n_col: int,
    row_ptr_a: NDArray[np.int32],
    columns_a: NDArray[np.int32],
    data_a: NDArray[np.float32],
    row_ptr_b: NDArray[np.int32],
    columns_b: NDArray[np.int32],
    data_b: NDArray[np.float32],
    row_ptr_c: NDArray[np.int32],
    columns_c: NDArray[np.int32],
    data_c: NDArray[np.float32],
) -> None:
    """Calculate C = A @ B with A, B and C CSR sparse matrices

    Parameters
    ----------
    n_row
        The number of rows in the resulting matrix
    n_col
        The number of columns in the resulting matrix
    row_ptr_a
        The index pointer array of the A CSR matrix
    columns_a
        The column values of the A CSR matrix
    data_a
        The data values in the matrix A
    row_ptr_b
        The index pointer array of the B CSR matrix
    columns_b
        The column values of the B CSR matrix
    data_b
        The data values in the matrix B
    row_ptr_c
        The index pointer array of the to be calculated C CSR matrix
    columns_c
        The column values of the to be calculated C CSR matrix
    data_c
        The data values in the to be calculated matrix C

    Notes
    -----
    Numba implementation of the C++ version found in the Scipy source code
    sparse._sparsetools.csr_matmat. Benchmarking shows this implementation
    is about 10-12% slower.
    """
    next = np.full(n_col, fill_value=-1, dtype=np.int32)
    sums = np.zeros(n_col, dtype=np.float32)

    nnz = int32(0)

    row_ptr_c[0] = int32(0)

    for row_a in range(n_row):
        head = int32(-2)
        length = int32(0)

        for nze_index_a in range(row_ptr_a[row_a], row_ptr_a[row_a + 1]):
            col_a = columns_a[nze_index_a]
            value_a = data_a[nze_index_a]

            for nze_index_b in range(row_ptr_b[col_a], row_ptr_b[col_a + 1]):
                col_b = columns_b[nze_index_b]
                value_b = data_b[nze_index_b]

                sums[col_b] += value_a * value_b

                if next[col_b] == int32(-1):
                    next[col_b] = head
                    head = col_b
                    length += int32(1)

        for _ in range(length):
            if sums[head] != float32(0):
                columns_c[nnz] = head
                data_c[nnz] = sums[head]
                nnz += int32(1)

            temp = head
            head = next[head]

            next[temp] = int32(-1)  # clear arrays again
            sums[temp] = float32(0)

        row_ptr_c[row_a + 1] = nnz


@njit
def csr_matmat_maxnnz(
    n_row: int,
    n_col: int,
    row_ptr_a: NDArray[np.int32],
    columns_a: NDArray[np.int32],
    row_ptr_b: NDArray[np.int32],
    columns_b: NDArray[np.int32],
) -> int:
    """Return the maximum number of non zero elements in A @ B with A and B CSR sparse

    Parameters
    ----------
    n_row
        The number of rows in the resulting matrix
    n_col
        The number of columns in the resulting matrix
    row_ptr_a
        The index pointer array of the A CSR matrix
    columns_a
        The column values of the A CSR matrix
    row_ptr_b
        The index pointer array of the B CSR matrix
    columns_b
        The column values of the B CSR matrix

    Returns
    -------
    nnz
        The maximum number of non zero elements in the matrix multiplication

    Notes
    -----
    Numba implementation of the C++ version found in the Scipy source code.
    This implementation is about 70-80% slower!
    """
    mask = np.full(n_col, -1, dtype=np.int32)

    nnz = int32(0)

    for row_a in range(n_row):
        row_nnz = int32(0)

        for nze_index_a in range(row_ptr_a[row_a], row_ptr_a[row_a + 1]):
            column_a = columns_a[nze_index_a]

            for nze_index_b in range(row_ptr_b[column_a], row_ptr_b[column_a + 1]):
                column_b = columns_b[nze_index_b]

                if mask[column_b] != row_a:
                    mask[column_b] = row_a
                    row_nnz += int32(1)

        nnz += row_nnz

    return nnz


@njit
def csr_dot_vec(
    row_ptr: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
    b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Calculate the matrix product Ab where A is a sparse CSR matrix and b
    a dense column vector.
    """
    # About equally fast as scipy implementation
    n_rows = row_ptr.shape[0] - 1
    result = np.zeros(
        n_rows,
        dtype=np.float32,
    )
    # sum together segments
    for i in range(n_rows):
        for nnz_index in range(row_ptr[i], row_ptr[i + 1]):
            result[i] += b[columns[nnz_index]] * data[nnz_index]

    return result


@njit
def coo_to_csr(
    n_rows: int,
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> CSRMatrix:
    """Convert row, column, data to row_ptr, column, data.

    Notes
    -----
    It is not assumed that data is sorted.
    It is assumed there are no duplicates.
    """
    nnz = rows.shape[0]
    row_ptr = row_ptr_from_coo(n_rows, rows)

    columns_out = np.empty_like(columns)
    data_out = np.empty_like(data)

    for i in range(nnz):
        row = rows[i]
        new_idx = row_ptr[row]

        columns_out[new_idx] = columns[i]
        data_out[new_idx] = data[i]

        # Increment the pointer so the next element in the same row is written
        # to the next position
        row_ptr[row] += 1

    # Incrementation of the row pointer nnz times is the same as shifting all
    # elements to the left. Here we shift them back.
    row_ptr[1:] = row_ptr[:n_rows]
    row_ptr[0] = 0

    return row_ptr, columns_out, data_out


@njit
def coo_to_csr_sorted(
    n_rows: int,
    rows: NDArray[np.int32],
    columns: NDArray[np.int32],
    data: NDArray[np.float32],
) -> CSRMatrix:
    """Convert row, column, data to row_ptr, column, data.

    Notes
    -----
    It is assumed that data is sorted by (row, column).
    It is assumed there are no duplicates.
    """
    columns = columns.copy()
    data = data.copy()
    row_ptr = row_ptr_from_coo(n_rows, rows)

    return row_ptr, columns, data


@njit
def row_ptr_from_coo(
    n_rows: int,
    rows: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Convert an array of row coordinates to a row pointer array"""
    nnz = rows.shape[0]
    row_ptr = np.zeros(n_rows + 1, dtype=np.int32)

    # count up all values
    for i in range(nnz):
        row_ptr[rows[i] + 1] += 1

    # accumulate
    for row in range(n_rows):
        row_ptr[row + 1] += row_ptr[row]

    return row_ptr
