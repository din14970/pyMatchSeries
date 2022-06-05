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
# import sparseqr


class InterpolationBase:
    def __init__(self, data):
        self.grid_shape = np.asarray(data.shape, dtype=int32)

    def get_coeffs(self, pos):
        ind = np.floor(pos).astype(int32)
        c = pos-ind
        valid = np.array([True, True])
        for k in range(2):
            if pos[k] < 0:
                valid[k] = False
                ind[k] = 0
                c[k] = 0
            elif pos[k] > self.grid_shape[k]-1:
                valid[k] = False
                ind[k] = self.grid_shape[k]-2
                c[k] = 1
            elif ind[k] == self.grid_shape[k]-1:
                ind[k] -= 1
                c[k] = 1
        return valid, ind, c


spec = [
    ('grid_shape', int32[::1]),
    ('data', float32[:, ::1]),
]


@jitclass(spec)
class BilinearInterpolation(InterpolationBase):
    __init__base = InterpolationBase.__init__

    def __init__(self, data):
        self.__init__base(data)
        self.data = np.ascontiguousarray(data)

    def evaluate(self, pos):
        result = np.zeros(pos.shape[:2])
        num_0 = pos.shape[0]
        num_1 = pos.shape[1]
        for i in range(num_0):
            for j in range(num_1):
                valid, ind, c = self.get_coeffs(pos[i, j, :])
                result[i, j] = (
                    self.data[ind[0], ind[1]] * (1 - c[0]) * (1 - c[1])
                    + self.data[ind[0] + 1, ind[1]] * c[0] * (1 - c[1])
                    + self.data[ind[0], ind[1] + 1] * (1 - c[0]) * c[1]
                    + self.data[ind[0] + 1, ind[1] + 1] * c[0] * c[1]
                )
        return result

    def evaluate_gradient(self, pos):
        result = np.zeros(pos.shape)
        num_0 = pos.shape[0]
        num_1 = pos.shape[1]
        for i in range(num_0):
            for j in range(num_1):
                valid, ind, c = self.get_coeffs(pos[i, j, :])
                if valid[0]:
                    result[i, j, 0] = (self.data[ind[0]+1, ind[1]]-self.data[ind[0], ind[1]])*(1-c[1]) \
                        + (self.data[ind[0]+1, ind[1]+1] -
                           self.data[ind[0], ind[1]+1])*c[1]
                if valid[1]:
                    result[i, j, 1] = (self.data[ind[0], ind[1]+1]-self.data[ind[0], ind[1]])*(1-c[0]) \
                        + (self.data[ind[0]+1, ind[1]+1] -
                           self.data[ind[0]+1, ind[1]])*c[0]
        return result


@njit
def _eval_im_at_coords(im, x, y, default):
    """
    Return from an image a list of values at float coordinates x, y.
    Presumes linear interpolation.
    Similar to ndimage.map_coordinates.

    Parameters
    ----------
    im : 2D numpy array (W, H)
    x, y : 1D numpy array (L)
    default : float
        Value if the coordinate is outside the image

    Returns
    -------
    result : 1D numpy array (L)
    """
    i = y.astype(np.int32)
    j = x.astype(np.int32)
    wx1 = j + 1.0 - x
    wx2 = x - j
    wy1 = i + 1.0 - y
    wy2 = y - i
    result = np.empty(x.shape, dtype=np.float32)
    for c in range(result.shape[0]):
        xx1 = j[c]
        yy1 = i[c]
        xx2 = j[c] + 1
        yy2 = i[c] + 1
        im1 = default
        im2 = default
        im3 = default
        im4 = default
        if 0 <= xx1 < im.shape[1] and 0 <= yy1 < im.shape[0]:
            im1 = im[yy1, xx1]
        if 0 <= xx2 < im.shape[1] and 0 <= yy1 < im.shape[0]:
            im2 = im[yy1, xx2]
        if 0 <= xx1 < im.shape[1] and 0 <= yy2 < im.shape[0]:
            im3 = im[yy2, xx1]
        if 0 <= xx2 < im.shape[1] and 0 <= yy2 < im.shape[0]:
            im4 = im[yy2, xx2]
        result[c] = im1 * wy1[c] * wx1[c] + im2 * wy1[c] * wx2[c] + im3 * wy2[c] * wx1[c] + im4 * wy2[c] * wx2[c]
    return result


@njit
def _eval_imdx_at_coords(im, x, y, default):
    """
    Return from an image the derivative with respect to x at float coordinates x, y.
    Presumes linear interpolation.
    """
    i = y.astype(np.int32)
    j = x.astype(np.int32)
    wy1 = i + 1.0 - y
    wy2 = y - i
    result = np.empty(x.shape, dtype=np.float32)
    for c in range(result.shape[0]):
        xx1 = j[c]
        yy1 = i[c]
        xx2 = j[c] + 1
        yy2 = i[c] + 1
        im1 = default
        im2 = default
        im3 = default
        im4 = default
        if 0 <= xx1 < im.shape[1] and 0 <= yy1 < im.shape[0]:
            im1 = im[yy1, xx1]
        if 0 <= xx2 < im.shape[1] and 0 <= yy1 < im.shape[0]:
            im2 = im[yy1, xx2]
        if 0 <= xx1 < im.shape[1] and 0 <= yy2 < im.shape[0]:
            im3 = im[yy2, xx1]
        if 0 <= xx2 < im.shape[1] and 0 <= yy2 < im.shape[0]:
            im4 = im[yy2, xx2]
        result[c] = -im1 * wy1[c] + im2 * wy1[c] + -im3 * wy2[c] + im4 * wy2[c]
    return result


@njit
def _eval_imdy_at_coords(im, x, y, default):
    """
    Return from an image the derivative with respect to y at float coordinates x, y.
    Presumes linear interpolation.
    """
    i = y.astype(np.int32)
    j = x.astype(np.int32)
    wx1 = j + 1.0 - x
    wx2 = x - j
    result = np.empty(x.shape, dtype=np.float32)
    for c in range(result.shape[0]):
        xx1 = j[c]
        yy1 = i[c]
        xx2 = j[c] + 1
        yy2 = i[c] + 1
        im1 = default
        im2 = default
        im3 = default
        im4 = default
        if 0 <= xx1 < im.shape[1] and 0 <= yy1 < im.shape[0]:
            im1 = im[yy1, xx1]
        if 0 <= xx2 < im.shape[1] and 0 <= yy1 < im.shape[0]:
            im2 = im[yy1, xx2]
        if 0 <= xx1 < im.shape[1] and 0 <= yy2 < im.shape[0]:
            im3 = im[yy2, xx1]
        if 0 <= xx2 < im.shape[1] and 0 <= yy2 < im.shape[0]:
            im4 = im[yy2, xx2]
        result[c] = -im1 * wx1[c] + -im2 * wx2[c] + im3 * wx1[c] + im4 * wx2[c]
    return result


def _get_gauss_quad_points_2():
    """
    Get the x, y coordinates of the Gaussian quadrature points with 4 points
    """
    p = 1 / np.sqrt(3) / 2
    quads = np.array([[-p, -p], [p, -p], [-p, p], [p, p]], dtype=np.float32)
    quads += 0.5
    return quads


def _get_gauss_quad_weights_2():
    """
    Get the weights for the Gaussian quadrature points with 4 points
    """
    return np.ones(4, dtype=np.float32) / 4


def _get_gauss_quad_points_3():
    """
    Get the x, y coordinates of the Gaussian quadrature points with 9 points
    """
    p = np.sqrt(3 / 5) / 2
    quads = np.array([[-p, -p], [0, -p], [p, -p], [-p, 0], [0, 0], [p, 0], [-p, p], [0, p], [p, p]], dtype=np.float32)
    quads += 0.5
    return quads


def _get_gauss_quad_weights_3():
    """
    Get the weights for the Gaussian quadrature points with 9 points
    """
    # from http://users.metu.edu.tr/csert/me582/ME582%20Ch%2003.pdf
    return np.array([25, 40, 25, 40, 64, 40, 25, 40, 25], dtype=np.float32) / 81 / 4


def _get_node_weights(q_coords):
    """
    For x, y coordinates in [0, 1] get the weights w_i that each surrounding node contributes to
    evaluating the function at x, y: f(x,y) = w_0 * f_00 + w_1 * f_01 + w_2 * f_10 + w_4 * f_11.
    """
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    wx1 = 1 - qx
    wx2 = qx
    wy1 = 1 - qy
    wy2 = qy
    return np.vstack([wy1 * wx1, wy1 * wx2, wy2 * wx1, wy2 * wx2])


def _get_dx_node_weights(q_coords):
    """
    The weights to evaluate d/dx * f at x, y coordinates, see _get_node_weights
    """
    # derive _get_node_weights wrt x
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    return np.vstack([-(1 - qy), 1 - qy, -qy, qy])


def _get_dy_node_weights(q_coords):
    """
    The weights to evaluate d/dy * f at x, y coordinates, see _get_node_weights
    """
    # derive _get_node_weights wrt y
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    return np.vstack([-(1 - qx), -qx, 1 - qx, qx])


def _get_qv1(q_coords):
    """Basis function evaluated top left of node"""
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    return qx * qy


def _get_qv2(q_coords):
    """Basis function evaluated top right of node"""
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    return (1 - qx) * qy


def _get_qv3(q_coords):
    """Basis function evaluated bottom left of node"""
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    return (1 - qy) * qx


def _get_qv4(q_coords):
    """Basis function evaluated bottom right of node"""
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    return (1 - qx) * (1 - qy)


def _get_qv(q_coords):
    """Evaluate the four basis functions at adjacent corners of coordinates (x, y)"""
    qv = np.empty((4, q_coords.shape[0]), dtype=np.float32)
    qv[0, :] = _get_qv1(q_coords)
    qv[1, :] = _get_qv2(q_coords)
    qv[2, :] = _get_qv3(q_coords)
    qv[3, :] = _get_qv4(q_coords)
    return qv


def _get_dqv1(q_coords):
    """Basis function gradient evaluated top left of node (d/dx, d/dy)"""
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    return (qy, qx)


def _get_dqv2(q_coords):
    """Basis function gradient evaluated top right of node (d/dx, d/dy)"""
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    return (-qy, (1 - qx))


def _get_dqv3(q_coords):
    """Basis function gradient evaluated bottom left of node (d/dx, d/dy)"""
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    return ((1 - qy), -qx)


def _get_dqv4(q_coords):
    """Basis function gradient evaluated bottom right of node (d/dx, d/dy)"""
    qx = q_coords[:, 0]
    qy = q_coords[:, 1]
    return (-(1 - qy), -(1 - qx))


def _get_dqv(q_coords):
    """Evaluate the four basis function gradients at adjacent corners of coordinates (x, y)"""
    qv = np.empty((4, 2, q_coords.shape[0]), dtype=np.float32)
    qv[0, :, :] = _get_dqv1(q_coords)
    qv[1, :, :] = _get_dqv2(q_coords)
    qv[2, :, :] = _get_dqv3(q_coords)
    qv[3, :, :] = _get_dqv4(q_coords)
    return qv


@njit
def _value_at_quad_points(im, node_weights):
    """
    Evaluate the value of the image at each quadrature point. The weights that determine each corner's
    contribution to the value at the quadrature point should be calculated a priory with a function
    like _get_node_weights(_get_gauss_quad_points_i()).
    Returned is an array of shape ((im.shape[0]-1)*(im.shape[1]-1), node_weights.shape[1]) where the first
    dimension corresponds to the number of cells and the second to the number of quad points in each cell.
    """
    output = np.empty((im.shape[0] - 1, im.shape[1] - 1, node_weights.shape[1]), dtype=np.float32)
    for r in range(im.shape[0] - 1):
        for c in range(im.shape[1] - 1):
            for p in range(node_weights.shape[1]):
                output[r, c, p] = (
                    im[r, c] * node_weights[0, p]
                    + im[r, c + 1] * node_weights[1, p]
                    + im[r + 1, c] * node_weights[2, p]
                    + im[r + 1, c + 1] * node_weights[3, p]
                )
    return output.reshape(((im.shape[0] - 1) * (im.shape[1] - 1), node_weights.shape[1]))


def residual(disp_x, disp_y, im1_interp, im2, node_weights, quad_weights_sqrt, mat_reg_full, L_sqrt, identity, grid_h):
    # we evaluate integral_over_domain (f(phi(x)) - g(x))**2 where x are all quad points (x_i, y_i)
    # first we evaluate (phi_x, phy_y) and g(x) at all quad points
    pos_x = _value_at_quad_points(disp_x/grid_h + identity[1, ...], node_weights).ravel()
    pos_y = _value_at_quad_points(disp_y/grid_h + identity[0, ...], node_weights).ravel()
    g = _value_at_quad_points(im2, node_weights)
    # then we evaluate f(phi_x, phi_y)
    pos = np.stack((pos_y, pos_x), axis=-1)[np.newaxis, ...]
    f = im1_interp.evaluate(pos).reshape(-1, node_weights.shape[1])

    res_data = np.multiply(quad_weights_sqrt, (f - g))

    return np.concatenate((res_data.ravel(), mat_reg_full * np.concatenate((disp_y.ravel(), disp_x.ravel()))))


def energy(disp_x, disp_y, im1_interp, im2, node_weights, node_weights_dx, node_weights_dy, quad_weights, L, identity, grid_h):
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
    pos_x = _value_at_quad_points(disp_x/grid_h + identity[1, ...], node_weights).ravel()
    pos_y = _value_at_quad_points(disp_y/grid_h + identity[0, ...], node_weights).ravel()
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
    partial_deriv = np.zeros((2, quadeval.shape[1] + 1, quadeval.shape[2] + 1), dtype=np.float32)
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
                        quadeval[c, cy0, cx0] * qv[0] + dphi_dx[c, cy0, cx0] * dqvx[0] + dphi_dy[c, cy0, cx0] * dqvy[0],
                        quad_weights,
                    )
            if 0 <= cx1 < quadeval.shape[2] and 0 <= cy0 < quadeval.shape[1]:
                for c in range(2):
                    partial_deriv[c, i, j] += np.dot(
                        quadeval[c, cy0, cx1] * qv[1] + dphi_dx[c, cy0, cx1] * dqvx[1] + dphi_dy[c, cy0, cx1] * dqvy[1],
                        quad_weights,
                    )
            if 0 <= cx0 < quadeval.shape[2] and 0 <= cy1 < quadeval.shape[1]:
                for c in range(2):
                    partial_deriv[c, i, j] += np.dot(
                        quadeval[c, cy1, cx0] * qv[2] + dphi_dx[c, cy1, cx0] * dqvx[2] + dphi_dy[c, cy1, cx0] * dqvy[2],
                        quad_weights,
                    )
            if 0 <= cx1 < quadeval.shape[2] and 0 <= cy1 < quadeval.shape[1]:
                for c in range(2):
                    partial_deriv[c, i, j] += np.dot(
                        quadeval[c, cy1, cx1] * qv[3] + dphi_dx[c, cy1, cx1] * dqvx[3] + dphi_dy[c, cy1, cx1] * dqvy[3],
                        quad_weights,
                    )
    return partial_deriv


@njit()
def _integrate_pd_over_cells_single(quadeval, quad_weights, qv):
    partial_deriv = np.zeros((quadeval.shape[0] + 1, quadeval.shape[1] + 1), dtype=np.float32)

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
    """
    """
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
    grid_h
):
    # Evaluates d/dphi_k (E)
    # 1) integrate_over_domain 2*(f(phi(x)) - g(x)) * f'(phi(x)) * dphi/dphi_k
    # dphi/dphi_k = basis_function_k
    # a) phi_x(x), phi_y(x), g(x) at all quad coords
    pos_x = _value_at_quad_points(disp_x/grid_h + identity[1, ...], node_weights).ravel()
    pos_y = _value_at_quad_points(disp_y/grid_h + identity[0, ...], node_weights).ravel()
    g = _value_at_quad_points(im2, node_weights)
    # b) evaluate first part of integrand but not yet the 2* as it appears also later
    pos = np.stack((pos_y, pos_x), axis=-1)[np.newaxis, ...]
    f = im1_interp.evaluate(pos).reshape(-1, node_weights.shape[1]).astype(np.float32)
    two_f_min_g = f - g
    # c) evaluate f' at phi_x, phi_y
    df = im1_interp.evaluate_gradient(pos)/grid_h
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
    grid_h
):
    pos_x = _value_at_quad_points(disp_x/grid_h + identity[1, ...], node_weights)
    pos_y = _value_at_quad_points(disp_y/grid_h + identity[0, ...], node_weights)
    pos = np.stack((pos_y, pos_x), axis=-1)
    cell_shape = (disp_x.shape[0] - 1, disp_x.shape[1] - 1, node_weights.shape[1])
    df = im1_interp.evaluate_gradient(pos)/grid_h
    dfdy = df[..., 0].reshape(cell_shape).astype(np.float32)
    dfdx = df[..., 1].reshape(cell_shape).astype(np.float32)
    data_y, rows_y, cols_y = _evaluate_pd_on_quad_points(dfdy, quad_weights_sqrt, qv)
    data_x, rows_x, cols_x = _evaluate_pd_on_quad_points(dfdx, quad_weights_sqrt, qv)

    mat_data = csr_matrix((np.concatenate((data_y, data_x)),
                           (np.concatenate((rows_y, rows_x)),
                            np.concatenate((cols_y, cols_x+disp_x.size)))
                          ), shape=(pos_x.size, 2*disp_x.size))

    return vstack([mat_data, mat_reg_full])


class RegistrationObjectiveFunction:

    # Constants, only calculate once on module import
    Q_POINTS = _get_gauss_quad_points_3()
    Q_WEIGHTS = _get_gauss_quad_weights_3()
    NODE_WEIGHTS = _get_node_weights(Q_POINTS)
    DX_NODE_WEIGHTS = _get_dx_node_weights(Q_POINTS)
    DY_NODE_WEIGHTS = _get_dy_node_weights(Q_POINTS)
    QV = _get_qv(Q_POINTS)
    DQV = _get_dqv(Q_POINTS)

    def __init__(
        self,
        im1: NDArray[(Any, Any), Any],
        im2: NDArray[(Any, Any), Any],
        L: float,
    ):
        self.grid_shape = im1.shape
        self.grid_h = 1 / (max(self.grid_shape) - 1)

        # -------------------------------------------------------------
        # Quadrature point values that are used throughout to evaluate
        # functions and derivatives at the quad points
        self.node_weights = self.NODE_WEIGHTS
        # By scaling quadrature weights with the element volume, we normalize the integration domain so that
        # its longest axis has length 1.
        self.quad_weights = (self.grid_h ** 2) * self.Q_WEIGHTS
        # By decreasing the size of the elements, the size of the derivative increases.
        self.node_weights_dx = self.DX_NODE_WEIGHTS / self.grid_h
        self.node_weights_dy = self.DY_NODE_WEIGHTS / self.grid_h

        # -------------------------------------------------------------
        # Evaluation of basis function and derivative of basis function at quadrature points in 4 cells around a node
        self.qv = self.QV
        # By decreasing the size of the elements, the size of the derivative increases.
        dqv = self.DQV/self.grid_h
        self.dqvx = dqv[:, 0, :]
        self.dqvy = dqv[:, 1, :]

        self.im1_interp = BilinearInterpolation(im1)
        self.im2 = im2
        self.quad_weights_sqrt = np.sqrt(self.quad_weights)
        self.L_sqrt = np.sqrt(L)

        self.identity = np.mgrid[0.:im1.shape[0], 0.:im1.shape[1]].astype(np.float32)

        # The following only need to be stored for the "residual-free" implementation of the energy and its gradient
        self.L = L

        # The derivative of the regularizer is a constant matrix that we precompute here.
        quadeval = np.full(
            (self.grid_shape[0] - 1, self.grid_shape[1] - 1, self.node_weights.shape[1]),
            fill_value=self.L_sqrt,
            dtype=np.float32
        )
        data_reg_x, rows_reg_x, cols_reg_x = _evaluate_pd_on_quad_points(quadeval, self.quad_weights_sqrt, self.dqvx)
        data_reg_y, rows_reg_y, cols_reg_y = _evaluate_pd_on_quad_points(quadeval, self.quad_weights_sqrt, self.dqvy)
        mat_reg = csr_matrix(
            (
                np.concatenate((data_reg_x, data_reg_y)),
                (
                    np.concatenate((rows_reg_x, rows_reg_y + quadeval.size)),
                    np.concatenate((cols_reg_x, cols_reg_y)),
                ),
            ),
            shape=(2 * quadeval.size, self.im2.size),
        )

        mat_zero = csr_matrix((2*quadeval.size, self.im2.size))
        self.mat_reg_full = vstack([hstack([mat_zero, mat_reg]), hstack([mat_reg, mat_zero])])

    def evaluate_residual(self, disp_vec):
        disp = disp_vec.reshape((2,) + self.grid_shape)
        return residual(disp[1, ...], disp[0, ...], self.im1_interp, self.im2, self.node_weights, self.quad_weights_sqrt, self.mat_reg_full, self.L_sqrt, self.identity, self.grid_h)

    def evaluate_residual_gradient(self, disp_vec):
        disp = disp_vec.reshape((2,) + self.grid_shape)
        return residual_gradient(disp[1, ...], disp[0, ...], self.im1_interp, self.node_weights, self.quad_weights_sqrt, self.mat_reg_full, self.qv, self.identity, self.grid_h)

    def evaluate_energy(self, disp_vec):
        # disp = disp_vec.reshape((2,) + self.grid_shape)
        # return energy(disp[1, ...], disp[0, ...], self.im1_interp, self.im2, self.node_weights, self.node_weights_dx, self.node_weights_dy, self.quad_weights, self.L, self.identity, self.grid_h)
        return np.sum(self.evaluate_residual(disp_vec)**2)

    def evaluate_energy_gradient(self, disp_vec):
        # disp = disp_vec.reshape((2,) + self.grid_shape)
        # return gradient(disp[1, ...], disp[0, ...], self.im1_interp, self.im2, self.node_weights, self.node_weights_dx, self.node_weights_dy, self.quad_weights, self.qv, self.dqvx, self.dqvy, self.L, self.identity, self.grid_h).ravel()
        res = self.evaluate_residual(disp_vec)
        mat = self.evaluate_residual_gradient(disp_vec)
        return 2*mat.T*res.ravel()


def getTimestepWidthWithSimpleLineSearch(E, direction, x, start_tau=1, tau_min=2**-30):
    # extreme simple timestep width control, just ensures, that fnew < fnew

    tau = start_tau

    e = E(x)
    eNew = E(x+tau*direction)

    while ((eNew >= e) and (tau >= tau_min)):
        tau = tau * 0.5
        eNew = E(x+tau*direction)

    # No energy descent for tau >= tauMin, so we don't want the step to be done.
    # The stopping criterion also handles this case.
    if (tau < tau_min):
        tau = 0

    return tau


def GaussNewtonAlgorithm(x0, F, DF, maxIter=50, stopEpsilon=0):
    x = x0.copy()
    f = F(x)
    fNormSqrOld = np.linalg.norm(f)**2
    print("Initial fNormSqr {:#.6g}".format(fNormSqrOld))
    tau = 1

    # use tqdm to show a progress bar
    i_bar = tqdm(range(maxIter))
    for i in i_bar:
        matDF = DF(x)
        # Solve the linear least-squares sense using a Cholesky factorization of the normal equations.
        # Note it would better to directly assemble the transposed instead of assembling and then
        # transposing.
        A = matDF.T
        factor = cholesky_AAt(A)
        direction = factor.solve_A(A*f)

        if not np.all(np.isfinite(direction)):
            print("Error: lstsq failed.")

        x -= direction

        f = F(x)
        fNormSqr = np.linalg.norm(f)**2

        # If the target functional did not decrease with the update, try to find a smaller step
        # so that it does. This step size control is extremely simple and not very efficient, but
        # it's certainly better than letting the algorithm diverge.
        if fNormSqr >= fNormSqrOld:
            x += direction
            direction *= -1
            # getTimestepWidthWithSimpleLineSearch doesn't support "widening", so let it start with 2*tau.
            tau = getTimestepWidthWithSimpleLineSearch(
                lambda v: np.linalg.norm(F(v))**2, direction, x, start_tau=min(2*tau, 1))
            x += tau*direction
            f = F(x)
            fNormSqr = np.linalg.norm(f)**2
        else:
            tau = 1

        i_bar.set_description("\u03C4={:#.2g}, E={:#.5g}, \u0394={:.1e}".format(
            tau, fNormSqr, fNormSqrOld - fNormSqr))

        if ((fNormSqrOld - fNormSqr)) <= stopEpsilon * fNormSqr or np.isclose(fNormSqr, 0):
            break
        fNormSqrOld = fNormSqr

    return x


def main():
    # im1 and im2 are two images (float32 dtype) of the same size assumed to be available
    im1 = np.zeros((128, 128), dtype=np.float32)
    im2 = np.zeros((128, 128), dtype=np.float32)

    im1[2*10:2*30, 2*15:2*45] = 1
    im2[2*25:2*45, 2*25:2*55] = 1

    # Regularization parameter
    L = 0.1
    num_levels = 5

    # Create an image hierarchy for both of our images
    pyramid_tem = tuple(pyramid_gaussian(im1, max_layer=num_levels-1, downscale=2, channel_axis=None))
    pyramid_ref = tuple(pyramid_gaussian(im2, max_layer=num_levels-1, downscale=2, channel_axis=None))

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
            disp = np.stack([resize(disp_new[0, ...], image_tem.shape), resize(disp_new[1, ...], image_tem.shape)])

        disp_new = GaussNewtonAlgorithm(
            disp.ravel(),
            objective.evaluate_residual,
            objective.evaluate_residual_gradient
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
                [disp_new[0, ...]/objective.grid_h+objective.identity[0, ...],
                 disp_new[1, ...]/objective.grid_h+objective.identity[1, ...]]
            )
        )
        plt.suptitle(f"Result for resolution {image_tem.shape}")
        plt.show()


if __name__ == "__main__":
    main()
