from __future__ import annotations

from types import ModuleType
from typing import Callable, Iterator, Mapping, Tuple, Union

import dask.array as da
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.sparse as sparse

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndimage
    import cupyx.scipy.sparse as csparse
    import cupyx.scipy.sparse.linalg as clinalg

    CUPY_IS_INSTALLED = True
except ImportError:
    cp = None
    csparse = None
    clinalg = None
    CUPY_IS_INSTALLED = False


ArrayType = Union[sparse.spmatrix, csparse.spmatrix, np.ndarray, cp.ndarray]
DenseArrayType = Union[np.ndarray, cp.ndarray]
SparseMatrixType = Union[sparse.spmatrix, csparse.spmatrix]


def mean(images: DenseArrayType) -> DenseArrayType:
    """Calculate the mean of an image stack, stack dimension is axis 0"""
    dp = get_dispatcher(images)
    return dp.mean(images, axis=0)


def median(images: DenseArrayType) -> DenseArrayType:
    """Calculate the median of an image stack, stack dimension is axis 0"""
    dp = get_dispatcher(images)
    return dp.median(images, axis=0)


def to_host(array: DenseArrayType) -> np.ndarray:
    if CUPY_IS_INSTALLED and isinstance(array, cp.ndarray):
        return array.get()
    elif isinstance(array, np.ndarray):
        return array
    else:
        raise ValueError(f"Array type is {type(array)}, must be array.")


def to_device(array: DenseArrayType) -> cp.ndarray:
    if CUPY_IS_INSTALLED and isinstance(array, cp.ndarray):
        return array
    elif isinstance(array, np.ndarray):
        return cp.asarray(array)
    else:
        raise ValueError(f"Array type is {type(array)}, must be array.")


def get_array_type(
    array: Union[np.ndarray, cp.ndarray, da.Array],
) -> Tuple[ModuleType, bool]:
    """Returns the underlying dispatcher and whether an array is lazy"""
    is_lazy = False
    if isinstance(array, da.Array):
        first_chunk_slice = tuple(slice(chunk_dim[0]) for chunk_dim in array.chunks)
        array = da.compute(array[first_chunk_slice])
        is_lazy = True
    return get_dispatcher(array), is_lazy


def displacement_to_coordinates(
    displacement: DenseArrayType,
) -> DenseArrayType:
    dp = get_dispatcher(displacement)
    grid_shape = (displacement.shape[1], displacement.shape[2])
    scaling_factor = get_grid_scaling_factor(grid_shape)
    identity = dp.mgrid[0 : grid_shape[0], 0 : grid_shape[1]].astype(displacement.dtype)
    return displacement / scaling_factor + identity


def get_grid_scaling_factor(grid_shape: Tuple[int, int]) -> float:
    return 1 / (max(grid_shape) - 1)


def map_coordinates(
    image: DenseArrayType,
    displacement: DenseArrayType,
    **kwargs,
) -> DenseArrayType:
    """Deform"""
    dp = get_dispatcher(image)
    ndi = get_ndimage_module(dp)
    return ndi.map_coordinates(
        image,
        displacement,
        order=kwargs.pop("order", 1),
        **kwargs,
    )


def create_image_pyramid(
    image: DenseArrayType,
    n_levels: int,
    downscale_factor: float = 2.0,
    **kwargs,
) -> Iterator[DenseArrayType]:
    """Create an iterator of an image resized by a constant factor"""
    smallest_dimension = min(image.shape)
    dp = get_dispatcher(image)
    ndi = get_ndimage_module(dp)
    sf = 1 / downscale_factor
    if smallest_dimension * sf ** (n_levels - 1) < 2:
        raise ValueError("The image size is reduced too much.")
    for level in reversed(range(n_levels)):
        yield ndi.zoom(
            image,
            sf**level,
            order=kwargs.pop("order", 1),
            **kwargs,
        )


def resize_image_stack(
    image_stack: DenseArrayType,
    new_size: Tuple[int, int],
    **kwargs,
) -> DenseArrayType:
    """Resize image stack to a new size. It is assumed the stack axis is at index 0"""
    dp = get_dispatcher(image_stack)
    ndi = get_ndimage_module(dp)
    original_shape = image_stack.shape
    new_shape = (image_stack.shape[0], *new_size)
    zoom = tuple(new / original for new, original in zip(new_shape, original_shape))
    output = dp.empty(new_shape, image_stack.dtype)
    ndi.zoom(
        image_stack,
        zoom,
        output=output,
        order=kwargs.pop("order", 1),
        **kwargs,
    )
    return output


class OneValueCache(dict):
    # adapted from https://stackoverflow.com/questions/2437617
    def __init__(self):
        dict.__init__(self)

    def __setitem__(self, key, value):
        if key not in self:
            self.clear()
            dict.__setitem__(self, key, value)


def get_dispatcher(array: DenseArrayType) -> ModuleType:
    """Returns the correct dispatcher to work with an array"""
    if CUPY_IS_INSTALLED and isinstance(array, cp.ndarray):
        return cp
    elif isinstance(array, np.ndarray):
        return np
    else:
        raise ValueError(f"Array type is {type(array)}, must be array.")


def get_sparse_module(dispatcher: ModuleType) -> ModuleType:
    if dispatcher == cp:
        return csparse
    elif dispatcher == np:
        return sparse
    else:
        raise ValueError("Array must be numpy or cupy array")


def get_ndimage_module(dispatcher: ModuleType) -> ModuleType:
    if dispatcher == cp:
        return cndimage
    elif dispatcher == np:
        return ndimage
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

    def __init__(self, matrix: ArrayType) -> None:
        raise NotImplementedError("The array module could not be determined")

    @property
    def module(self) -> ModuleType:
        raise NotImplementedError("The array module could not be determined")

    @classmethod
    def new(cls, matrix: ArrayType) -> Matrix:
        matrix_type = cls.get_matrix_type(matrix)
        return matrix_type(matrix)

    @classmethod
    def get_matrix_type(cls, matrix: ArrayType) -> type[Matrix]:
        if CUPY_IS_INSTALLED and isinstance(matrix, cp.ndarray):
            return CupyMatrix
        elif isinstance(matrix, np.ndarray):
            return NumpyMatrix
        elif CUPY_IS_INSTALLED and isinstance(matrix, csparse.spmatrix):
            return SparseCupyMatrix
        elif isinstance(matrix, sparse.spmatrix):
            return SparseNumpyMatrix
        else:
            raise ValueError(f"Array type is {type(matrix)}, must be array.")

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
        return np

    def to_device(self) -> CupyMatrix:
        if not CUPY_IS_INSTALLED:
            raise RuntimeError("Cupy must be installed.")
        return CupyMatrix(cp.array(self.data))

    def to_sparse(self, sparse_type: str = "coo") -> SparseNumpyMatrix:
        sparse_array = self._to_sparse_methods_cpu[sparse_type](self.data)
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
        return scipy.linalg.solve(MtM, Mtb, assume_a="pos")


class CupyMatrix(Matrix):
    def __init__(self, matrix: cp.ndarray) -> None:
        self.data = matrix

    @property
    def module(self) -> ModuleType:
        raise cp

    def to_host(self) -> NumpyMatrix:
        return NumpyMatrix(cp.asnumpy(self.data))

    def to_sparse(self, sparse_type: str = "coo") -> SparseCupyMatrix:
        sparse_array = self._to_sparse_methods_gpu[sparse_type](self.data)
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

        else:
            raise RuntimeError("Unrecognized sparse matrix format.")

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

        else:
            raise RuntimeError("Unrecognized sparse matrix format.")

    def solve(self, b: cp.ndarray) -> cp.ndarray:
        return clinalg.spsolve(self.data, b)

    def solve_lstsq(self, b: cp.ndarray) -> cp.ndarray:
        """Solve linear least squares directly for predictable performance"""
        M = self.data
        M_t = M.T
        MtM = M_t.dot(M)
        Mtb = M_t.dot(b)
        return clinalg.spsolve(MtM, Mtb)
