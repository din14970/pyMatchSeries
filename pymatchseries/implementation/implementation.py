import numpy as np
from nptyping import NDArray
from typing import Any, Sequence

from numba import njit, prange, int32, float32
from numba.experimental import jitclass
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, zoom
from scipy.optimize import minimize, least_squares
from scipy.sparse import csr_matrix, vstack, hstack, linalg
from skimage.transform import pyramid_gaussian, resize
from tqdm import tqdm
from sksparse.cholmod import cholesky_AAt


from .interpolation import BilinearInterpolation2D









@njit
def _value_at_quad_points(im, node_weights):
    """
    Evaluate the value of the image at each quadrature point. The weights that determine each corner's
    contribution to the value at the quadrature point should be calculated a priory with a function
    like _get_node_weights(_get_gauss_quad_points_i()).
    Returned is an array of shape ((im.shape[0]-1)*(im.shape[1]-1), node_weights.shape[1]) where the first
    dimension corresponds to the number of cells and the second to the number of quad points in each cell.
    """
    output = np.empty(
        (im.shape[0] - 1, im.shape[1] - 1, node_weights.shape[1]), dtype=np.float32
    )
    for r in range(im.shape[0] - 1):
        for c in range(im.shape[1] - 1):
            for p in range(node_weights.shape[1]):
                output[r, c, p] = (
                    im[r, c] * node_weights[0, p]
                    + im[r, c + 1] * node_weights[1, p]
                    + im[r + 1, c] * node_weights[2, p]
                    + im[r + 1, c + 1] * node_weights[3, p]
                )
    return output.reshape(
        ((im.shape[0] - 1) * (im.shape[1] - 1), node_weights.shape[1])
    )


def residual(
    disp_x,
    disp_y,
    im1_interp,
    im2,
    node_weights,
    quad_weights_sqrt,
    mat_reg_full,
    L_sqrt,
    identity,
    grid_h,
):
    # we evaluate integral_over_domain (f(phi(x)) - g(x))**2 where x are all quad points (x_i, y_i)
    # first we evaluate (phi_x, phy_y) and g(x) at all quad points
    pos_x = _value_at_quad_points(
        disp_x / grid_h + identity[1, ...], node_weights
    ).ravel()
    pos_y = _value_at_quad_points(
        disp_y / grid_h + identity[0, ...], node_weights
    ).ravel()
    g = _value_at_quad_points(im2, node_weights)
    # then we evaluate f(phi_x, phi_y)
    pos = np.stack((pos_y, pos_x), axis=-1)[np.newaxis, ...]
    f = im1_interp.evaluate(pos).reshape(-1, node_weights.shape[1])

    res_data = np.multiply(quad_weights_sqrt, (f - g))

    return np.concatenate(
        (
            res_data.ravel(),
            mat_reg_full * np.concatenate((disp_y.ravel(), disp_x.ravel())),
        )
    )


def energy(
    disp_x,
    disp_y,
    im1_interp,
    im2,
    node_weights,
    node_weights_dx,
    node_weights_dy,
    quad_weights,
    L,
    identity,
    grid_h,
):
    """
    The function that should be minimized

    Parameters
    ----------
    phi_x : (H, W) numpy array
        The x component of the phi field
    phi_y : (H, W) numpy array
        The y component of the phi field
    im1 : (H, W) numpy array
        The first image f
    im2 : (H, W) numpy array
        The second image g
    node_weights : (4, P) numpy array
        The weights of each of the 4 vertices of a cell for each P quadrature points
        to evaluate the function in that point
    node_weights_dx : (4, P) numpy array
        The weights of the 4 vertices of a cell for each P quadrature points
        to evaluate df/dx in that point
    node_weights_dy : (4, P) numpy array
        The weights of the 4 vertices of a cell for each P quadrature points
        to evaluate df/dy in that point
    quad_weights : (P) numpy array
        The weights of each quadrature point when evaluating an integral over
        the domain
    L : float
        Regularization constant
    """
    # we evaluate integral_over_domain (f(phi(x)) - g(x))**2 where x are all quad points (x_i, y_i)
    # first we evaluate (phi_x, phy_y) and g(x) at all quad points
    pos_x = _value_at_quad_points(
        disp_x / grid_h + identity[1, ...], node_weights
    ).ravel()
    pos_y = _value_at_quad_points(
        disp_y / grid_h + identity[0, ...], node_weights
    ).ravel()
    g = _value_at_quad_points(im2, node_weights)
    # then we evaluate f(phi_x, phi_y)
    pos = np.stack((pos_y, pos_x), axis=-1)[np.newaxis, ...]
    f = im1_interp.evaluate(pos).reshape(-1, node_weights.shape[1])
    # we evaluate integral with Gaussian quadrature = multiply integrand by weights of quad points and sum
    integrated = np.dot(np.sum((f - g) ** 2, axis=0), quad_weights)
    # regularization term = integral_over_domain of
    # (dphi_x/dx - 1)**2 + (dphi_x/dy)**2 + (dphi_y/dx)**2 + (dphi_y/dy-1)**2
    # The same formula can be used but plugging in different node weights accounting for differentiation
    disp_x_dx = _value_at_quad_points(disp_x, node_weights_dx)
    disp_y_dx = _value_at_quad_points(disp_y, node_weights_dx)
    disp_x_dy = _value_at_quad_points(disp_x, node_weights_dy)
    disp_y_dy = _value_at_quad_points(disp_y, node_weights_dy)
    # the same integration trick
    regxx = np.dot(np.sum((disp_x_dx) ** 2, axis=0), quad_weights)
    regyx = np.dot(np.sum((disp_y_dx) ** 2, axis=0), quad_weights)
    regxy = np.dot(np.sum((disp_x_dy) ** 2, axis=0), quad_weights)
    regyy = np.dot(np.sum((disp_y_dy) ** 2, axis=0), quad_weights)
    return integrated + L * (regxx + regyx + regxy + regyy)


@njit()
def _integrate_pd_over_cells(
    quadeval,
    quad_weights,
    dphi_dx,
    dphi_dy,
    qv,
    dqvx,
    dqvy,
):
    """
    Evaluate the partial derivative of E with respect to all phi_k by summing over the cells

    Parameters
    ----------
    quadeval : (2, cells_y, cells_x, P) array
        The dense part of the integral evaluated at each P quad point in cells_y * cells_x cells.
        quadeval[0] is the y-component and quadeval[1] is the x-component
    quad_weights : (P) array
        integration weights to each quad point
    dphi_dx: (2, cells_y, cells_x, P) array
        The regularisation term contribution from the derivative of phi wrt x. The component of phi
        should match the component quadeval pertains to.
        dphi_dx[0] is the y-component (dphiy_dx) and quadeval[1] (dphix_dx) is the x-component
    dphi_dy: (cells_y, cells_x, P) array
        The regularisation term contribution from the derivative of phi wrt y. The component of phi
        should match the component quadeval pertains to.
        dphi_dy[0] is the y-component (dphiy_dy) and quadeval[1] (dphix_dy) is the x-component
    qvi : (P) array with P
        value of the k'th basis function at the quad points in the i'th quadrant
    dqvix : (P) array with P
        value of the derivative of k'th basis function wrt x at the quad points with i denoting the quadrant
    dqviy : (P) array with P
        value of the derivative of k'th basis function wrt y at the quad points with i denoting the quadrant

    Returns
    -------
    partial_deriv : (2, cells_y + 1, cells_x + 1) array
        The partial derivative of E at each node. partial_deriv[0] is wrt y, partial_deriv[1] is wrt x

    Notes
    -----
    The quadrant numbering convention we use are:
    * 1: top left from node
    * 2: top right from node
    * 3: bottom left from node
    * 4: bottom right from node
    """
    # we have an extra row and column of nodes to evaluate partial derivative at
    partial_deriv = np.zeros(
        (2, quadeval.shape[1] + 1, quadeval.shape[2] + 1), dtype=np.float32
    )
    # we loop over the nodes but must translate this into cell coordinates
    for i in prange(quadeval.shape[1] + 1):
        for j in range(quadeval.shape[2] + 1):
            cx0 = j - 1
            cx1 = j
            cy0 = i - 1
            cy1 = i
            # add the contributions from the 4 neighboring cells around a node
            if 0 <= cx0 < quadeval.shape[2] and 0 <= cy0 < quadeval.shape[1]:
                for c in range(2):
                    partial_deriv[c, i, j] += np.dot(
                        quadeval[c, cy0, cx0] * qv[0]
                        + dphi_dx[c, cy0, cx0] * dqvx[0]
                        + dphi_dy[c, cy0, cx0] * dqvy[0],
                        quad_weights,
                    )
            if 0 <= cx1 < quadeval.shape[2] and 0 <= cy0 < quadeval.shape[1]:
                for c in range(2):
                    partial_deriv[c, i, j] += np.dot(
                        quadeval[c, cy0, cx1] * qv[1]
                        + dphi_dx[c, cy0, cx1] * dqvx[1]
                        + dphi_dy[c, cy0, cx1] * dqvy[1],
                        quad_weights,
                    )
            if 0 <= cx0 < quadeval.shape[2] and 0 <= cy1 < quadeval.shape[1]:
                for c in range(2):
                    partial_deriv[c, i, j] += np.dot(
                        quadeval[c, cy1, cx0] * qv[2]
                        + dphi_dx[c, cy1, cx0] * dqvx[2]
                        + dphi_dy[c, cy1, cx0] * dqvy[2],
                        quad_weights,
                    )
            if 0 <= cx1 < quadeval.shape[2] and 0 <= cy1 < quadeval.shape[1]:
                for c in range(2):
                    partial_deriv[c, i, j] += np.dot(
                        quadeval[c, cy1, cx1] * qv[3]
                        + dphi_dx[c, cy1, cx1] * dqvx[3]
                        + dphi_dy[c, cy1, cx1] * dqvy[3],
                        quad_weights,
                    )
    return partial_deriv


@njit()
def _integrate_pd_over_cells_single(quadeval, quad_weights, qv):
    partial_deriv = np.zeros(
        (quadeval.shape[0] + 1, quadeval.shape[1] + 1), dtype=np.float32
    )

    # Iterate over all cells
    for i in prange(quadeval.shape[0]):
        for j in range(quadeval.shape[1]):
            partial_deriv[i + 1, j + 1] += np.dot(quadeval[i, j] * qv[0], quad_weights)
            partial_deriv[i + 1, j] += np.dot(quadeval[i, j] * qv[1], quad_weights)
            partial_deriv[i, j + 1] += np.dot(quadeval[i, j] * qv[2], quad_weights)
            partial_deriv[i, j] += np.dot(quadeval[i, j] * qv[3], quad_weights)

    return partial_deriv


@njit()
def ravel_index(pos: Sequence[int], shape: Sequence[int]):
    """
    Get the index of an array element if that array was flattened
    """
    # Adapted from https://stackoverflow.com/a/4271004
    res = 0
    acc = 1
    for pi, si in zip(pos[::-1], shape[::-1]):
        res += pi * acc
        acc *= si
    return res


@njit()
def _evaluate_pd_on_quad_points(quadeval, quad_weights_sqrt, dqv):
    """ """
    num = 4 * quadeval.size
    data = np.zeros(num, dtype=np.float32)
    rows = np.floor(np.arange(0, quadeval.size, 0.25))
    cols = np.zeros(num, dtype=np.float32)
    idx = 0
    # original data shape
    dof_shape = (quadeval.shape[0] + 1, quadeval.shape[1] + 1)
    for i in prange(quadeval.shape[0]):
        for j in range(quadeval.shape[1]):
            for k in range(quadeval.shape[2]):
                val = quadeval[i, j, k] * quad_weights_sqrt[k]

                data[idx] = val * dqv[0, k]
                cols[idx] = ravel_index((i + 1, j + 1), dof_shape)
                idx = idx + 1

                data[idx] = val * dqv[1, k]
                cols[idx] = ravel_index((i + 1, j), dof_shape)
                idx = idx + 1

                data[idx] = val * dqv[2, k]
                cols[idx] = ravel_index((i, j + 1), dof_shape)
                idx = idx + 1

                data[idx] = val * dqv[3, k]
                cols[idx] = ravel_index((i, j), dof_shape)
                idx = idx + 1

    return data, rows, cols


def gradient(
    disp_x,
    disp_y,
    im1_interp,
    im2,
    node_weights,
    node_weights_dx,
    node_weights_dy,
    quad_weights,
    qv,
    dqvx,
    dqvy,
    L,
    identity,
    grid_h,
):
    # Evaluates d/dphi_k (E)
    # 1) integrate_over_domain 2*(f(phi(x)) - g(x)) * f'(phi(x)) * dphi/dphi_k
    # dphi/dphi_k = basis_function_k
    # a) phi_x(x), phi_y(x), g(x) at all quad coords
    pos_x = _value_at_quad_points(
        disp_x / grid_h + identity[1, ...], node_weights
    ).ravel()
    pos_y = _value_at_quad_points(
        disp_y / grid_h + identity[0, ...], node_weights
    ).ravel()
    g = _value_at_quad_points(im2, node_weights)
    # b) evaluate first part of integrand but not yet the 2* as it appears also later
    pos = np.stack((pos_y, pos_x), axis=-1)[np.newaxis, ...]
    f = im1_interp.evaluate(pos).reshape(-1, node_weights.shape[1]).astype(np.float32)
    two_f_min_g = f - g
    # c) evaluate f' at phi_x, phi_y
    df = im1_interp.evaluate_gradient(pos) / grid_h
    dfdy = df[..., 0].reshape(-1, node_weights.shape[1]).astype(np.float32)
    dfdx = df[..., 1].reshape(-1, node_weights.shape[1]).astype(np.float32)
    # d) multiply
    cell_shape = (disp_x.shape[0] - 1, disp_x.shape[1] - 1, node_weights.shape[1])
    prodx = (two_f_min_g * dfdx).reshape(cell_shape)
    prody = (two_f_min_g * dfdy).reshape(cell_shape)
    # 2) regularization term is 2*(dphi_x/dx - 1)* d(basis_function_k)/d_x + 2*(dphi_x/dy)* d(basis_func_k)/dy
    # and                       2*(dphi_y/dx)* d(basis_func_k)/dx + 2*(dphi_y/dy - 1)* d(basis_function_k)/dy
    disp_x_dx = _value_at_quad_points(disp_x, node_weights_dx).reshape(cell_shape)
    disp_y_dx = _value_at_quad_points(disp_y, node_weights_dx).reshape(cell_shape)
    disp_x_dy = _value_at_quad_points(disp_x, node_weights_dy).reshape(cell_shape)
    disp_y_dy = _value_at_quad_points(disp_y, node_weights_dy).reshape(cell_shape)
    # 3) integrate over all the cells
    partial_y = _integrate_pd_over_cells_single(2 * prody, quad_weights, qv)
    partial_x = _integrate_pd_over_cells_single(2 * prodx, quad_weights, qv)
    partial_y += _integrate_pd_over_cells_single(2 * L * disp_y_dx, quad_weights, dqvx)
    partial_y += _integrate_pd_over_cells_single(2 * L * disp_y_dy, quad_weights, dqvy)
    partial_x += _integrate_pd_over_cells_single(2 * L * disp_x_dx, quad_weights, dqvx)
    partial_x += _integrate_pd_over_cells_single(2 * L * disp_x_dy, quad_weights, dqvy)

    return np.stack((partial_y, partial_x))


def residual_gradient(
    disp_x,
    disp_y,
    im1_interp,
    node_weights,
    quad_weights_sqrt,
    mat_reg_full,
    qv,
    identity,
    grid_h,
):
    pos_x = _value_at_quad_points(disp_x / grid_h + identity[1, ...], node_weights)
    pos_y = _value_at_quad_points(disp_y / grid_h + identity[0, ...], node_weights)
    pos = np.stack((pos_y, pos_x), axis=-1)
    cell_shape = (disp_x.shape[0] - 1, disp_x.shape[1] - 1, node_weights.shape[1])
    df = im1_interp.evaluate_gradient(pos) / grid_h
    dfdy = df[..., 0].reshape(cell_shape).astype(np.float32)
    dfdx = df[..., 1].reshape(cell_shape).astype(np.float32)
    data_y, rows_y, cols_y = _evaluate_pd_on_quad_points(dfdy, quad_weights_sqrt, qv)
    data_x, rows_x, cols_x = _evaluate_pd_on_quad_points(dfdx, quad_weights_sqrt, qv)

    mat_data = csr_matrix(
        (
            np.concatenate((data_y, data_x)),
            (
                np.concatenate((rows_y, rows_x)),
                np.concatenate((cols_y, cols_x + disp_x.size)),
            ),
        ),
        shape=(pos_x.size, 2 * disp_x.size),
    )

    return vstack([mat_data, mat_reg_full])




def main():
    # im1 and im2 are two images (float32 dtype) of the same size assumed to be available
    im1 = np.zeros((128, 128), dtype=np.float32)
    im2 = np.zeros((128, 128), dtype=np.float32)

    im1[2 * 10 : 2 * 30, 2 * 15 : 2 * 45] = 1
    im2[2 * 25 : 2 * 45, 2 * 25 : 2 * 55] = 1

    # Regularization parameter
    L = 0.1
    num_levels = 5

    # Create an image hierarchy for both of our images
    pyramid_tem = tuple(
        pyramid_gaussian(im1, max_layer=num_levels - 1, downscale=2, channel_axis=None)
    )
    pyramid_ref = tuple(
        pyramid_gaussian(im2, max_layer=num_levels - 1, downscale=2, channel_axis=None)
    )

    disp_new = None

    for i in reversed(range(num_levels)):
        image_tem = pyramid_tem[i]
        image_ref = pyramid_ref[i]

        objective = RegistrationObjectiveFunction(image_tem, image_ref, L)

        # initialize displacement as zero if we no guess from a previous level
        if disp_new is None:
            disp = np.zeros_like(objective.identity)
        # initialize displacement by upsampling the one from the previous level
        else:
            disp = np.stack(
                [
                    resize(disp_new[0, ...], image_tem.shape),
                    resize(disp_new[1, ...], image_tem.shape),
                ]
            )

        disp_new = GaussNewtonAlgorithm(
            disp.ravel(),
            objective.evaluate_residual,
            objective.evaluate_residual_gradient,
        ).reshape(disp.shape)
        # res = minimize(objective.evaluate_energy, disp.ravel(), jac=objective.evaluate_energy_gradient, method="BFGS", options={"disp": True, "maxiter": 1000})
        # disp_new = res.x.reshape(disp.shape)

        mpl.rcParams["image.cmap"] = "gray"
        _, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].title.set_text("tem")
        ax[0].imshow(image_tem)
        ax[1].title.set_text("ref")
        ax[1].imshow(image_ref)
        ax[2].title.set_text("tem(phi)")
        ax[2].imshow(
            map_coordinates(
                image_tem,
                [
                    disp_new[0, ...] / objective.grid_h + objective.identity[0, ...],
                    disp_new[1, ...] / objective.grid_h + objective.identity[1, ...],
                ],
            )
        )
        plt.suptitle(f"Result for resolution {image_tem.shape}")
        plt.show()


if __name__ == "__main__":
    main()
