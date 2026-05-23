from types import ModuleType
from typing import Tuple

import numpy as np
import pytest
from skimage.transform import pyramid_gaussian

from pymatchseries.implementation.objective_functions import (
    RegistrationObjectiveFunction,
)
from pymatchseries.implementation.solvers import root_gauss_newton
from pymatchseries.utils import CUPY_IS_INSTALLED, DenseArrayType, cp

if CUPY_IS_INSTALLED:
    params = [np, cp]
else:
    params = [np]


def _setup(dp: ModuleType = np) -> Tuple[RegistrationObjectiveFunction, DenseArrayType]:
    im1 = np.zeros((128, 128), dtype=np.float32)
    im2 = np.zeros((128, 128), dtype=np.float32)

    im1[2 * 10 : 2 * 30, 2 * 15 : 2 * 45] = 1
    im2[2 * 25 : 2 * 45, 2 * 25 : 2 * 55] = 1

    num_levels = 5

    # Create an image hierarchy for both of our images
    pyramid_tem = tuple(
        pyramid_gaussian(im1, max_layer=num_levels - 1, downscale=2, channel_axis=None)
    )
    pyramid_ref = tuple(
        pyramid_gaussian(im2, max_layer=num_levels - 1, downscale=2, channel_axis=None)
    )

    # Regularization parameter
    L = 0.1

    image_def = pyramid_tem[-1]
    image_ref = pyramid_ref[-1]

    if dp != np:
        image_def = cp.asarray(image_def)
        image_ref = cp.asarray(image_ref)

    objective = RegistrationObjectiveFunction(
        image_ref,
        image_def,
        L,
    )

    disp = dp.zeros_like(objective.identity)
    return objective, disp


@pytest.mark.parametrize("dp", params)
def test_root_gauss_newton(dp: ModuleType) -> None:
    objective, disp = _setup(dp)
    disp_new = root_gauss_newton(
        objective.evaluate_residual,
        disp.ravel(),
        objective.evaluate_residual_gradient,
    ).reshape(disp.shape)

    assert objective.evaluate_energy(disp.ravel()) > objective.evaluate_energy(
        disp_new.ravel()
    )
