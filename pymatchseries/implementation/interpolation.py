from typing import Tuple
from numba import int32, float32
from numba.experimental import jitclass
import numpy as np


try:
    import cupy as cp
except ImportError:
    cp = None


class InterpolationBase2D:
    def __init__(
        self,
        image: np.ndarray,
    ):
        self.height = image.shape[0]
        self.width = image.shape[1]

    def _get_interpolation_data(
        self,
        coordinate: float,
        axis_size: int,
    ) -> Tuple[bool, int, float]:
        reference_gripoint = int(coordinate)
        if coordinate < 0:
            is_valid = False
            reference_gridpoint = 0
            weight = 0.
        elif coordinate >= axis_size - 1:
            is_valid = False
            reference_gridpoint = axis_size - 2
            weight = 1.
        else:
            is_valid = True
            weight = coordinate - reference_gripoint
        return is_valid, reference_gridpoint, weight


spec = [
    ("height", int32),
    ("width", int32),
    ("image", float32[:, ::1]),
]


@jitclass(spec)
class BilinearInterpolation2D(InterpolationBase2D):
    __init__base = InterpolationBase2D.__init__

    def __init__(
        self,
        image: np.ndarray,
    ):
        self.__init__base(image)
        self.image = np.ascontiguousarray(image)

    def evaluate(self, coordinates: np.ndarray) -> np.ndarray:
        result = np.zeros(coordinates.shape[:2])
        rows = coordinates.shape[0]
        columns = coordinates.shape[1]
        for row in range(rows):
            for column in range(columns):
                y = coordinates[row, column, 0]
                x = coordinates[row, column, 1]
                _, y0, wy = self._get_interpolation_data(y, self.height)
                _, x0, wx = self._get_interpolation_data(x, self.width)
                result[row, column] = (
                    self.image[y0, x0] * (1 - wy) * (1 - wx)
                    + self.image[y0 + 1, x0] * wy * (1 - wx)
                    + self.image[y0, x0 + 1] * (1 - wy) * wx
                    + self.image[y0 + 1, x0 + 1] * wy * wx
                )
        return result

    def evaluate_gradient(self, coordinates: np.ndarray) -> np.ndarray:
        result = np.zeros(coordinates.shape)
        rows = coordinates.shape[0]
        columns = coordinates.shape[1]
        for row in range(rows):
            for column in range(columns):
                y = coordinates[row, column, 0]
                x = coordinates[row, column, 1]
                valid_y, y0, wy = self._get_interpolation_data(y, self.height)
                valid_x, x0, wx = self._get_interpolation_data(x, self.width)
                if valid_y:
                    result[row, column, 0] = (
                        (self.image[y0 + 1, x0] - self.image[y0, x0]) * (1 - wy)
                        + (self.image[y0 + 1, x0 + 1] - self.image[y0, x0 + 1]) * wx
                    )
                if valid_x:
                    result[row, column, 1] = (
                        (self.image[y0, x0 + 1] - self.image[y0, x0]) * (1 - wy)
                        + (self.image[y0 + 1, x0 + 1] - self.image[y0 + 1, x0]) * wx
                    )
        return result
