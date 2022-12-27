import numpy as np
from types import ModuleType
from pymatchseries.implementation.interpolation import (
    interpolate_cpu,
    interpolate_gradient_cpu,
)
from pymatchseries.implementation.cuda_kernels import (
    cp,
    interpolate_gpu,
    interpolate_gradient_gpu,
)


def _get_image_and_coords(dispatcher: ModuleType = np):
    dispatcher.random.seed(42)
    image = dispatcher.array(
        [
            [2, 5, 7, 8, 1],
            [3, 1, 2, 0, 0],
            [1, 1, 0, 4, 6],
            [1, 2, 1, 0, 5],
        ]
    ).astype(dispatcher.float32)
    coordinates = dispatcher.mgrid[0: image.shape[0], 0: image.shape[1]]
    coordinates = dispatcher.moveaxis(coordinates, 0, -1)
    coordinates = coordinates.astype(dispatcher.float32)
    jitter = dispatcher.random.rand(*coordinates.shape) - 0.5
    coordinates_2 = coordinates + jitter
    return (
        image,
        coordinates,
        coordinates_2,
    )


def test_interpolate_cpu():
    image, coordinates, coordinates_2 = _get_image_and_coords()
    result = interpolate_cpu(image, coordinates)
    np.testing.assert_allclose(image, result)
    result_2 = interpolate_cpu(image, coordinates_2)
    expected = np.array(
        [
            [3.3521428, 4.2464533, 6.3119893, 5.436767, 0.898885],
            [2.707175, 1.3840603, 3.173547, 1.5321381, 0.16751026],
            [1.0, 1.0555547, 1.1533972, 2.818614, 4.8747425],
            [1.0, 1.116166, 0.69160265, 0.79082614, 4.7007623],
        ]
    )
    np.testing.assert_allclose(expected, result_2)


def test_interpolate_gradient_cpu():
    image, coordinates, coordinates_2 = _get_image_and_coords()
    result = interpolate_gradient_cpu(image, coordinates)
    verify = np.array(
        [
            [[1.0, 3.0], [-4.0, 2.0], [-5.0, 1.0], [-8.0, -7.0], [-1.0, -7.0]],
            [[-2.0, -2.0], [0.0, 1.0], [-2.0, -2.0], [4.0, 0.0], [6.0, 0.0]],
            [[0.0, 0.0], [1.0, -1.0], [1.0, 4.0], [-4.0, 2.0], [-1.0, 2.0]],
            [[0.0, 1.0], [1.0, -1.0], [1.0, -1.0], [-4.0, 5.0], [-1.0, 5.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(verify, result)
    result_2 = interpolate_gradient_cpu(image, coordinates_2)
    verify = np.array(
        [
            [
                [0.0, 3.0],
                [-4.0986586, 1.7680061],
                [0.0, 2.0],
                [0.0, -7.0],
                [-1.0, 0.0],
            ],
            [
                [-1.3495493, 0.39707753],
                [-0.5753218, -1.3351147],
                [-4.6834044, 1.3181751],
                [-7.826705, -1.3703043],
                [-2.461396, -0.47638488],
            ],
            [
                [0.0, 0.0],
                [-0.26727632, -0.41571072],
                [-0.28894424, 3.73642],
                [4.028469, 1.3993475],
                [-2.3606489, 2.2772436],
            ],
            [
                [0.0, 0.0],
                [1.0, -1.0],
                [0.0, -1.0],
                [-1.9883605, -0.02306885],
                [0.0, 5.0],
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(verify, result_2, rtol=5e-7)


def test_interpolate_gpu():
    image, coordinates, coordinates_2 = _get_image_and_coords(cp)
    result = interpolate_gpu(image, coordinates)
    cp.testing.assert_allclose(image, result)
    result_2 = interpolate_gpu(image, coordinates_2)
    expected = cp.array(
        [
            [2.3351169, 4.784593, 6.323781, 7.888076, 0.6047023],
            [2.4068134, 1.0830803, 1.5158693, 1.6066545, 2.2851026],
            [1.0, 1.0497671, 1.8376315, 3.6594667, 5.994196],
            [1.0, 1.8726237, 1.1656399, 0.21499483, 3.532943],
        ],
        dtype=cp.float32,
    )
    cp.testing.assert_allclose(expected, result_2)


def test_interpolate_gradient_gpu():
    image, coordinates, coordinates_2 = _get_image_and_coords(cp)
    result = interpolate_gradient_gpu(image, coordinates)
    verify = cp.array(
        [
            [[1.0, 3.0], [-4.0, 2.0], [-5.0, 1.0], [-8.0, -7.0], [-1.0, -7.0]],
            [[-2.0, -2.0], [0.0, 1.0], [-2.0, -2.0], [4.0, 0.0], [6.0, 0.0]],
            [[0.0, 0.0], [1.0, -1.0], [1.0, 4.0], [-4.0, 2.0], [-1.0, 2.0]],
            [[0.0, 1.0], [1.0, -1.0], [1.0, -1.0], [-4.0, 5.0], [-1.0, 5.0]],
        ],
        dtype=cp.float32,
    )
    cp.testing.assert_allclose(cp.array(verify), result)
    result_2 = interpolate_gradient_gpu(image, coordinates_2)
    verify = cp.array(
        [
            [
                [0.48179293, 2.748931],
                [0.0, 3.0],
                [0.0, 2.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            [
                [-2.0, 0.0],
                [-0.6195262, 0.2682058],
                [-1.702734, 0.6059305],
                [-7.639736, -1.4721166],
                [6.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.6556361, 0.07590657],
                [-1.4056768, 3.6908605],
                [-3.909028, 3.6575165],
                [-1.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [1.0, -1.0],
                [0.0, -1.0],
                [0.0, 5.0],
                [-1.9338055, 4.8614874],
            ],
        ],
        dtype=cp.float32,
    )
    cp.testing.assert_allclose(verify, result_2)
