import numpy as np

from pymatchseries.implementation.implementation import JNRR, JNNRConfig


def test_multilevel() -> None:
    shape = (128, 132)
    im1 = np.zeros(shape, dtype=np.float32)
    im2 = np.zeros(shape, dtype=np.float32)

    im1[2 * 10 : 2 * 30, 2 * 15 : 2 * 45] = 1
    im2[2 * 25 : 2 * 45, 2 * 25 : 2 * 55] = 1

    configuration = JNNRConfig()
    configuration.solver.show_progress = False
    configuration.n_levels = 5
    L = 0.1

    displacement = JNRR._get_multilevel_displacement_field(
        image_deformed=im1,
        image_reference=im2,
        regularization_constant=L,
        configuration=configuration,
    )

    assert displacement.shape == (2, *shape)
