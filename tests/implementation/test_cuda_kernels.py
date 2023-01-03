import numpy as np
import pytest

from pymatchseries.implementation.cuda_kernels import (
    evaluate_at_quad_points_gpu,
    evaluate_pd_on_quad_points_gpu,
    interpolate_gpu,
    interpolate_gradient_gpu,
)
from pymatchseries.implementation.interpolation import (
    interpolate_cpu,
    interpolate_gradient_cpu,
)
from pymatchseries.implementation.quadrature import (
    evaluate_at_quad_points_cpu,
    evaluate_pd_on_quad_points_cpu,
)
from pymatchseries.utils import CUPY_IS_INSTALLED, cp

RTOL = 1e-6


@pytest.mark.skipif(
    not CUPY_IS_INSTALLED,
    reason="cupy not installed, gpu probably not installed",
)
def test_interpolate_gpu() -> None:
    image = np.random.rand(400, 500).astype(np.float32)
    coordinates = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    coordinates = np.moveaxis(coordinates, 0, -1)
    coordinates = coordinates.astype(np.float32)
    jitter = np.random.rand(*coordinates.shape) - 0.5
    coordinates_2 = coordinates + jitter

    image_gpu = cp.asarray(image)
    coordinates_gpu = cp.asarray(coordinates)
    coordinates_2_gpu = cp.asarray(coordinates_2)

    result_cpu_1 = interpolate_cpu(image, coordinates)
    result_gpu_1 = interpolate_gpu(image_gpu, coordinates_gpu)

    result_cpu_2 = interpolate_cpu(image, coordinates_2)
    result_gpu_2 = interpolate_gpu(image_gpu, coordinates_2_gpu)

    np.testing.assert_allclose(cp.asnumpy(result_gpu_1), result_cpu_1, rtol=RTOL)
    np.testing.assert_allclose(cp.asnumpy(result_gpu_2), result_cpu_2, rtol=RTOL)


@pytest.mark.skipif(
    not CUPY_IS_INSTALLED,
    reason="cupy not installed, gpu probably not installed",
)
def test_interpolate_gradient_gpu() -> None:
    image = np.random.rand(400, 500).astype(np.float32)
    coordinates = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    coordinates = np.moveaxis(coordinates, 0, -1)
    coordinates = coordinates.astype(np.float32)
    jitter = np.random.rand(*coordinates.shape) - 0.5
    coordinates_2 = coordinates + jitter

    image_gpu = cp.asarray(image)
    coordinates_gpu = cp.asarray(coordinates)
    coordinates_2_gpu = cp.asarray(coordinates_2)

    result_cpu_1 = interpolate_gradient_cpu(image, coordinates)
    result_gpu_1 = interpolate_gradient_gpu(image_gpu, coordinates_gpu)

    result_cpu_2 = interpolate_gradient_cpu(image, coordinates_2)
    result_gpu_2 = interpolate_gradient_gpu(image_gpu, coordinates_2_gpu)

    np.testing.assert_allclose(cp.asnumpy(result_gpu_1), result_cpu_1, rtol=RTOL)
    np.testing.assert_allclose(cp.asnumpy(result_gpu_2), result_cpu_2, rtol=RTOL)


@pytest.mark.skipif(
    not CUPY_IS_INSTALLED,
    reason="cupy not installed, gpu probably not installed",
)
def test_evaluate_at_quad_points_gpu() -> None:
    image = np.random.rand(400, 500).astype(np.float32)
    node_weights = np.random.rand(4, 11).astype(np.float32)

    image_gpu = cp.asarray(image)
    node_weights_gpu = cp.asarray(node_weights)

    result_cpu = evaluate_at_quad_points_cpu(image, node_weights)
    result_gpu = evaluate_at_quad_points_gpu(image_gpu, node_weights_gpu)

    np.testing.assert_allclose(cp.asnumpy(result_gpu), result_cpu, rtol=RTOL)


@pytest.mark.skipif(
    not CUPY_IS_INSTALLED,
    reason="cupy not installed, gpu probably not installed",
)
def test_evaluate_pd_on_quad_points_gpu() -> None:
    K = 11
    N = 400
    M = 500
    quadrature_values = np.random.rand(N - 1, M - 1, K).astype(np.float32)
    quad_weights_sqrt = np.random.rand(K).astype(np.float32)
    node_weights = np.random.rand(4, K).astype(np.float32)

    quadrature_values_gpu = cp.asarray(quadrature_values)
    quad_weights_sqrt_gpu = cp.asarray(quad_weights_sqrt)
    node_weights_gpu = cp.asarray(node_weights)

    result_cpu = evaluate_pd_on_quad_points_cpu(
        quadrature_values,
        quad_weights_sqrt,
        node_weights,
    )
    result_gpu = evaluate_pd_on_quad_points_gpu(
        quadrature_values_gpu,
        quad_weights_sqrt_gpu,
        node_weights_gpu,
    )

    np.testing.assert_allclose(cp.asnumpy(result_gpu[0]), result_cpu[0], rtol=RTOL)
    np.testing.assert_allclose(cp.asnumpy(result_gpu[1]), result_cpu[1], rtol=RTOL)
    np.testing.assert_allclose(cp.asnumpy(result_gpu[2]), result_cpu[2], rtol=RTOL)
