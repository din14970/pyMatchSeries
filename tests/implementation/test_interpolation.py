import numpy as np

from pymatchseries.implementation.interpolation import (
    interpolate_cpu,
    interpolate_gradient_cpu,
)

RTOL = 1e-6


def _get_image_and_coords():
    np.random.seed(42)
    image = np.array(
        [
            [2, 5, 7, 8, 1],
            [3, 1, 2, 0, 0],
            [1, 1, 0, 4, 6],
            [1, 2, 1, 0, 5],
        ]
    ).astype(np.float32)
    coordinates = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    coordinates = np.moveaxis(coordinates, 0, -1)
    coordinates = coordinates.astype(np.float32)
    # random array to be added to coordinates
    jitter = np.array(
        [
            [
                [-0.12545988, 0.45071431],
                [0.23199394, 0.09865848],
                [-0.34398136, -0.34400548],
                [-0.44191639, 0.36617615],
                [0.10111501, 0.20807258],
            ],
            [
                [-0.47941551, 0.46990985],
                [0.33244264, -0.28766089],
                [-0.31817503, -0.31659549],
                [-0.19575776, 0.02475643],
                [-0.06805498, -0.20877086],
            ],
            [
                [0.11185289, -0.36050614],
                [-0.20785535, -0.13363816],
                [-0.04393002, 0.28517596],
                [-0.30032622, 0.01423444],
                [0.09241457, -0.45354959],
            ],
            [
                [0.10754485, -0.32947588],
                [-0.43494841, 0.44888554],
                [0.46563203, 0.30839735],
                [-0.19538623, -0.40232789],
                [0.18423303, -0.05984751],
            ],
        ]
    )
    coordinates_2 = coordinates + jitter
    return (
        image,
        coordinates,
        coordinates_2,
    )


def test_interpolate_cpu():
    image, coordinates, coordinates_2 = _get_image_and_coords()
    result = interpolate_cpu(image, coordinates)

    result_2 = interpolate_cpu(image, coordinates_2)
    expected = np.array(
        [
            [3.3521428, 4.2464533, 6.3119893, 5.436767, 0.898885],
            [2.707175, 1.3840603, 3.173547, 1.5321381, 0.16751026],
            [1.0, 1.0555547, 1.1533972, 2.818614, 4.8747425],
            [1.0, 1.116166, 0.69160265, 0.79082614, 4.7007623],
        ]
    )

    np.testing.assert_allclose(image, result, rtol=RTOL)
    np.testing.assert_allclose(expected, result_2, rtol=RTOL)


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

    result_2 = interpolate_gradient_cpu(image, coordinates_2)
    verify_2 = np.array(
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

    np.testing.assert_allclose(verify, result, rtol=RTOL)
    np.testing.assert_allclose(verify_2, result_2, rtol=RTOL)
