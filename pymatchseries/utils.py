from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Mapping, Tuple, Union

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
    cndimage = None
    csparse = None
    clinalg = None
    CUPY_IS_INSTALLED = False


from numpy.typing import NDArray

if TYPE_CHECKING:
    Matrix = Union[sparse.spmatrix, np.ndarray, csparse.spmatrix, cp.ndarray]
    SparseMatrix = Union[sparse.spmatrix, csparse.spmatrix]
else:
    ArrayType = None
    SparseMatrix = None


def mean(images: NDArray) -> NDArray:
    """Calculate the mean of an image stack, stack dimension is axis 0"""
    return np.mean(images, axis=0)


def median(images: NDArray) -> NDArray:
    """Calculate the median of an image stack, stack dimension is axis 0"""
    return np.median(images, axis=0)


def to_host(array: NDArray) -> np.ndarray:
    """Get an array as a numpy array"""
    if CUPY_IS_INSTALLED and isinstance(array, cp.ndarray):
        return array.get()
    elif isinstance(array, np.ndarray):
        return array
    else:
        raise ValueError(f"Array type is {type(array)}, must be array.")


def to_device(array: NDArray) -> cp.ndarray:
    """Get an array as a cupy array"""
    if CUPY_IS_INSTALLED and isinstance(array, cp.ndarray):
        return array
    elif isinstance(array, np.ndarray):
        return cp.asarray(array)
    else:
        raise ValueError(f"Array type is {type(array)}, must be array.")


def to_device_sparse(matrix: sparse.spmatrix) -> csparse.spmatrix:
    """Get a sparse matrix represented on the device"""
    if not CUPY_IS_INSTALLED:
        raise RuntimeError("Cupy must be installed.")

    if isinstance(matrix, csparse.spmatrix):
        return matrix

    data = cp.asarray(matrix.data)

    if sparse.isspmatrix_coo(matrix):
        row = cp.asarray(matrix.row)
        col = cp.asarray(matrix.col)
        return csparse.coo_matrix((data, (row, col)))

    elif sparse.isspmatrix_csr(matrix):
        indices = cp.asarray(matrix.indices)
        indptr = cp.asarray(matrix.indptr)
        return csparse.csr_matrix((data, indices, indptr))

    elif sparse.isspmatrix_csc(matrix):
        indices = cp.asarray(matrix.indices)
        indptr = cp.asarray(matrix.indptr)
        return csparse.csc_matrix((data, indices, indptr))

    else:
        raise RuntimeError("Unrecognized sparse matrix format.")


def to_host_sparse(matrix: csparse.spmatrix) -> sparse.spmatrix:
    """Get a sparse matrix represented on the host"""
    if isinstance(matrix, sparse.spmatrix):
        return matrix

    data = cp.asnumpy(matrix.data)

    if csparse.isspmatrix_coo(matrix):
        row = cp.asnumpy(matrix.row)
        col = cp.asnumpy(matrix.col)
        return sparse.coo_matrix((data, (row, col)))

    elif csparse.isspmatrix_csr(matrix):
        indices = cp.asnumpy(matrix.indices)
        indptr = cp.asnumpy(matrix.indptr)
        return sparse.csr_matrix((data, indices, indptr))

    elif csparse.isspmatrix_csc(matrix):
        indices = cp.asnumpy(matrix.indices)
        indptr = cp.asnumpy(matrix.indptr)
        return sparse.csc_matrix((data, indices, indptr))

    else:
        raise RuntimeError("Unrecognized sparse matrix format.")


def displacement_to_coordinates(
    displacement: NDArray,
    scaling_factor: float,
    identity: NDArray,
) -> NDArray:
    """Convert a 2D displacement field to coordinates that when
    mapped on an image apply the displacement field
    """
    return displacement / scaling_factor + identity


def get_grid_scaling_factor(
    height: int,
    width: int,
) -> float:
    """Scale of the integration domain"""
    return 1 / (max(height, width) - 1)


def get_coordinate_identity(
    displacement: NDArray,
) -> NDArray:
    """Coordinates that match 0 displacement"""
    dp = get_dispatcher(displacement)
    height, width = displacement.shape[1], displacement.shape[2]
    return dp.mgrid[0:height, 0:width].astype(displacement.dtype)


def _map_coordinates(
    image: NDArray,
    coordinates: NDArray,
    output: NDArray,
    method: Callable,
    **kwargs,
) -> NDArray:
    """Map the input array to new coordinates by interpolation"""
    return method(
        image,
        coordinates,
        output=output,
        order=kwargs.pop("order", 1),
        **kwargs,
    )


def map_coordinates_cpu(
    image: NDArray,
    coordinates: NDArray,
    output: NDArray,
    **kwargs,
) -> np.ndarray:
    """Map the input array to new coordinates by interpolation on the CPU"""
    return _map_coordinates(
        image=image,
        coordinates=coordinates,
        output=output,
        method=ndimage.map_coordinates,
        **kwargs,
    )


def map_coordinates_gpu(
    image: NDArray,
    coordinates: NDArray,
    output: NDArray,
    **kwargs,
) -> np.ndarray:
    """Map the input array to new coordinates by interpolation on the GPU"""
    return _map_coordinates(
        image=image,
        coordinates=coordinates,
        output=output,
        method=cndimage.map_coordinates,
        **kwargs,
    )


def _create_image_pyramid(
    image: NDArray,
    n_levels: int,
    method: Callable,
    downscale_factor: float = 2.0,
    **kwargs,
) -> Iterator[NDArray]:
    """Create an iterator of an image resized by a constant factor"""
    smallest_dimension = min(image.shape)
    sf = 1 / downscale_factor
    if smallest_dimension * sf ** (n_levels - 1) < 2:
        raise ValueError("The image size is reduced too much.")
    for level in reversed(range(n_levels)):
        yield method(
            image,
            sf**level,
            order=kwargs.pop("order", 1),
            **kwargs,
        )


def create_image_pyramid_cpu(
    image: np.ndarray,
    n_levels: int,
    downscale_factor: float = 2.0,
    **kwargs,
) -> Iterator[np.ndarray]:
    """Create an iterator of an image resized by a constant factor"""
    return _create_image_pyramid(
        image=image,
        n_levels=n_levels,
        method=ndimage.zoom,
        downscale_factor=downscale_factor,
        **kwargs,
    )


def create_image_pyramid_gpu(
    image: cp.ndarray,
    n_levels: int,
    downscale_factor: float = 2.0,
    **kwargs,
) -> Iterator[cp.ndarray]:
    """Create an iterator of an image resized by a constant factor"""
    return _create_image_pyramid(
        image=image,
        n_levels=n_levels,
        method=cndimage.zoom,
        downscale_factor=downscale_factor,
        **kwargs,
    )


def _resize_image_stack(
    image_stack: NDArray,
    new_size: Tuple[int, int],
    method: Callable,
    **kwargs: Any,
) -> NDArray:
    """Resize image stack to a new size. It is assumed the stack axis is at index 0"""
    original_shape = image_stack.shape
    new_shape = (image_stack.shape[0], *new_size)
    zoom = tuple(new / original for new, original in zip(new_shape, original_shape))
    return method(
        image_stack,
        zoom,
        order=kwargs.pop("order", 1),
        **kwargs,
    )


def resize_image_stack_cpu(
    image_stack: np.ndarray,
    new_size: Tuple[int, int],
    **kwargs: Any,
) -> NDArray:
    """Resize image stack to a new size. It is assumed the stack axis is at index 0"""
    return _resize_image_stack(
        image_stack=image_stack,
        new_size=new_size,
        method=ndimage.zoom,
        **kwargs,
    )


def resize_image_stack_gpu(
    image_stack: cp.ndarray,
    new_size: Tuple[int, int],
    **kwargs: Any,
) -> NDArray:
    """Resize image stack to a new size. It is assumed the stack axis is at index 0"""
    return _resize_image_stack(
        image_stack=image_stack,
        new_size=new_size,
        method=cndimage.zoom,
        **kwargs,
    )


class OneValueCache(dict):
    # adapted from https://stackoverflow.com/questions/2437617
    def __init__(self):
        dict.__init__(self)

    def __setitem__(self, key, value):
        if key not in self:
            self.clear()
            dict.__setitem__(self, key, value)


def get_dispatcher(array: NDArray) -> ModuleType:
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


def solve_lstsq_cpu(
    M: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Solve linear least squares on the CPU"""
    M_t = M.T
    MtM = M_t.dot(M)
    Mtb = M_t.dot(b)
    # MtM will always be a positive definite matrix
    return scipy.linalg.solve(MtM, Mtb, assume_a="pos")


def solve_lstsq_gpu(
    M: cp.ndarray,
    b: cp.ndarray,
) -> cp.ndarray:
    """Solve linear least squares on the GPU"""
    M_t = M.T
    MtM = M_t.dot(M)
    Mtb = M_t.dot(b)
    return cp.linalg.solve(MtM, Mtb)


def solve_sparse_lstsq_cpu(
    M: sparse.spmatrix,
    b: np.ndarray,
) -> np.ndarray:
    """Solve sparse linear least squares on the CPU"""
    M_t = M.T
    MtM = M_t.dot(M)
    Mtb = M_t.dot(b)
    return sparse.linalg.spsolve(MtM, Mtb)


def solve_sparse_lstsq_gpu(
    M: csparse.spmatrix,
    b: cp.ndarray,
) -> cp.ndarray:
    """Solve sparse linear least squares on the GPU"""
    M_t = M.T
    MtM = M_t.dot(M)
    Mtb = M_t.dot(b)
    return clinalg.spsolve(MtM, Mtb)
