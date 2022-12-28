from __future__ import annotations
from typing import TypeVar, Mapping, Callable, Type
from types import ModuleType

import numpy as np
import scipy
import scipy.sparse as sparse

try:
    import cupy as cp
    import cupyx.scipy.sparse as csparse
    import cupyx.scipy.linalg as clinalg
    CUPY_IS_INSTALLED = True
except ImportError:
    cp = None
    csparse = None
    clinalg = None
    CUPY_IS_INSTALLED = False


ArrayType = TypeVar("ArrayType", sparse.spmatrix, csparse.spmatrix, np.ndarray, cp.ndarray)
DenseArrayType = TypeVar("DenseArrayType", np.ndarray, cp.ndarray)
SparseMatrixType = TypeVar("SparseMatrixType", sparse.spmatrix, csparse.spmatrix)


def get_dispatcher(array: DenseArrayType) -> ModuleType:
    """Returns the correct dispatcher to work with an array"""
    if CUPY_IS_INSTALLED and isinstance(array, cp.ndarray):
        return cp
    elif isinstance(array, np.ndarray):
        return np
    else:
        raise ValueError(f"Array type is {type(array)}, must be {ArrayType}.")


def get_sparse_module(dispatcher: ModuleType) -> ModuleType:
    if dispatcher == cp:
        return csparse
    elif dispatcher == np:
        return sparse
    else:
        raise ValueError("Array must be numpy or cupy array")


class Matrix:

    _to_sparse_methods_cpu: Mapping[str, Callable] = {
        "coo": sparse.coo_matrix,
        "csc": sparse.csc_matrix,
        "csr": sparse.csr_matrix,
    }

    _to_sparse_methods_gpu: Mapping[str, Callable] = {
        "coo": csparse.coo_matrix,
        "csc": csparse.csc_matrix,
        "csr": csparse.csr_matrix,
    }

    @property
    def module(self) -> ModuleType:
        raise NotImplementedError("The array module could not be determined")

    @classmethod
    def new(cls, matrix: ArrayType) -> Matrix:
        matrix_type = cls.get_matrix_type(matrix)
        return matrix_type(matrix)

    @classmethod
    def get_matrix_type(cls, matrix: ArrayType) -> Type[Matrix]:
        if CUPY_IS_INSTALLED and isinstance(matrix, cp.ndarray):
            return CupyMatrix
        elif isinstance(matrix, np.ndarray):
            return NumpyMatrix
        elif CUPY_IS_INSTALLED and isinstance(matrix, csparse.spmatrix):
            return SparseCupyMatrix
        elif isinstance(matrix, sparse.spmatrix):
            return SparseNumpyMatrix
        else:
            raise ValueError(f"Array type is {type(matrix)}, must be {ArrayType}.")

    def to_host(self) -> Matrix:
        return self

    def to_device(self) -> Matrix:
        return self

    def to_sparse(self, sparse_type: str) -> Matrix:
        return self

    def to_dense(self) -> Matrix:
        return self

    def solve(self, b: ArrayType) -> ArrayType:
        raise NotImplementedError("No solving method is implemented")

    def solve_lstsq(self, b: ArrayType) -> ArrayType:
        raise NotImplementedError("No least squares method is implemented")


class NumpyMatrix(Matrix):

    def __init__(self, matrix: np.ndarray) -> None:
        self.data = matrix

    @property
    def module(self) -> ModuleType:
        raise np

    def to_device(self) -> CupyMatrix:
        if not CUPY_IS_INSTALLED:
            raise RuntimeError("Cupy must be installed.")
        return CupyMatrix(cp.array(self.data))

    def to_sparse(self, sparse_type: str = "coo") -> SparseNumpyMatrix:
        sparse_array = self._to_sparse_methods[sparse_type](self.data)
        return SparseNumpyMatrix(sparse_array)

    def solve(self, b: np.ndarray) -> np.ndarray:
        return scipy.linalg.solve(self.data, b)

    def solve_lstsq(self, b: np.ndarray) -> np.ndarray:
        """Solve linear least squares directly for predictable performance"""
        M = self.data
        M_t = M.T
        MtM = M_t.dot(M)
        Mtb = M_t.dot(b)
        # MtM will always be a positive definite matrix
        return scipy.linalg.solve(MtM, Mtb, assume_a='pos')


class CupyMatrix(Matrix):

    def __init__(self, matrix: cp.ndarray) -> None:
        self.data = matrix

    @property
    def module(self) -> ModuleType:
        raise cp

    def to_host(self) -> NumpyMatrix:
        return NumpyMatrix(cp.asnumpy(self.data))

    def to_sparse(self, sparse_type: str = "coo") -> SparseNumpyMatrix:
        sparse_array = self._to_sparse_methods[sparse_type](self.data)
        return SparseCupyMatrix(sparse_array)

    def solve(self, b: np.ndarray) -> np.ndarray:
        return cp.linalg.solve(self.data, b)

    def solve_lstsq(self, b: np.ndarray) -> np.ndarray:
        """Solve linear least squares directly for predictable performance"""
        M = self.data
        M_t = M.T
        MtM = M_t.dot(M)
        Mtb = M_t.dot(b)
        return cp.linalg.solve(MtM, Mtb)


class SparseNumpyMatrix(Matrix):

    def __init__(self, matrix: sparse.spmatrix) -> None:
        self.data = matrix

    def to_dense(self) -> NumpyMatrix:
        return NumpyMatrix(self.data.toarray())

    def convert_to_csc(self) -> None:
        self.data = sparse.csc_matrix(self.data)

    def convert_to_csr(self) -> None:
        self.data = sparse.csr_matrix(self.data)

    def convert_to_coo(self) -> None:
        self.data = sparse.coo_matrix(self.data)

    def convert_to(self, sparse_type: str) -> None:
        self.data = self._to_sparse_methods_cpu[sparse_type](self.data)

    def to_device(self) -> SparseCupyMatrix:
        if not CUPY_IS_INSTALLED:
            raise RuntimeError("Cupy must be installed.")
        matrix = self.data
        data = cp.array(matrix.data)

        if sparse.isspmatrix_coo(matrix):
            row = cp.array(matrix.row)
            col = cp.array(matrix.col)
            return SparseCupyMatrix(csparse.coo_matrix((data, (row, col))))

        elif sparse.isspmatrix_csr(matrix):
            indices = cp.array(matrix.indices)
            indptr = cp.array(matrix.indptr)
            return SparseCupyMatrix(csparse.csr_matrix((data, indices, indptr)))

        elif sparse.isspmatrix_csc(matrix):
            indices = cp.array(matrix.indices)
            indptr = cp.array(matrix.indptr)
            return SparseCupyMatrix(csparse.csc_matrix((data, indices, indptr)))

    def solve(self, b: np.ndarray) -> np.ndarray:
        return sparse.linalg.spsolve(self.data, b)

    def solve_lstsq(self, b: np.ndarray) -> np.ndarray:
        """Solve linear least squares directly for predictable performance"""
        M = self.data
        M_t = M.T
        MtM = M_t.dot(M)
        Mtb = M_t.dot(b)
        return sparse.linalg.spsolve(MtM, Mtb)


class SparseCupyMatrix(Matrix):

    def __init__(self, matrix: csparse.spmatrix) -> None:
        self.data = matrix

    def to_dense(self) -> CupyMatrix:
        return CupyMatrix(self.data.toarray())

    def convert_to_csc(self) -> None:
        self.data = csparse.csc_matrix(self.data)

    def convert_to_csr(self) -> None:
        self.data = csparse.csr_matrix(self.data)

    def convert_to_coo(self) -> None:
        self.data = csparse.coo_matrix(self.data)

    def convert_to(self, sparse_type: str) -> None:
        self.data = self._to_sparse_methods_gpu[sparse_type](self.data)

    def to_host(self) -> SparseNumpyMatrix:
        matrix = self.data
        data = cp.asnumpy(matrix.data)

        if csparse.isspmatrix_coo(matrix):
            row = cp.asnumpy(matrix.row)
            col = cp.asnumpy(matrix.col)
            return SparseNumpyMatrix(sparse.coo_matrix((data, (row, col))))

        elif csparse.isspmatrix_csr(matrix):
            indices = cp.asnumpy(matrix.indices)
            indptr = cp.asnumpy(matrix.indptr)
            return SparseNumpyMatrix(sparse.csr_matrix((data, indices, indptr)))

        elif csparse.isspmatrix_csc(matrix):
            indices = cp.asnumpy(matrix.indices)
            indptr = cp.asnumpy(matrix.indptr)
            return SparseNumpyMatrix(sparse.csc_matrix((data, indices, indptr)))

    def solve(self, b: cp.ndarray) -> cp.ndarray:
        return clinalg.spsolve(self.data, b)

    def solve_lstsq(self, b: cp.ndarray) -> cp.ndarray:
        """Solve linear least squares directly for predictable performance"""
        M = self.data
        M_t = M.T
        MtM = M_t.dot(M)
        Mtb = M_t.dot(b)
        return clinalg.spsolve(MtM, Mtb)
