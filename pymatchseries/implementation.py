import numpy as np
from numba import njit


@njit
def _eval_im_at_coords(im, x, y, default):
    """
    Return from an image a list of values at float coordinates x, y. 
    Presumes linear interpolation.
    Similar to ndimage.map_coordinates but works only on flat x, y.
    
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
    wx1 = j + 1. - x
    wx2 = x - j
    wy1 = i + 1. - y
    wy2 = y - i
    result = np.zeros(x.shape, dtype=np.float32)
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
        result[c] = (im1*wy1[c]*wx1[c] + 
                     im2*wy1[c]*wx2[c] + 
                     im3*wy2[c]*wx1[c] + 
                     im4*wy2[c]*wx2[c])
    return result


def _get_gauss_quad_points_2():
    """
    Get the x, y coordinates of the Gaussian quadrature points with 4 points
    """
    p = 1/np.sqrt(3)/2
    quads = np.array([[-p, -p],
                      [p, -p],
                      [-p, p],
                      [p, p]],
                     dtype=np.float32
                    )
    quads += 0.5
    return quads


def _get_gauss_quad_weights_2():
    """
    Get the weights for the Gaussian quadrature points with 4 points
    """
    return np.ones(4, dtype=np.float32)


def _get_gauss_quad_points_3():
    """
    Get the x, y coordinates of the Gaussian quadrature points with 9 points
    """
    p = np.sqrt(3/5)/2
    quads = np.array([[-p, -p],
                      [0, -p],
                      [p, -p],
                      [-p, 0],
                      [0, 0],
                      [p, 0],
                      [-p, p],
                      [0, p],
                      [p, p]],
                     dtype=np.float32
                    )
    quads += 0.5
    return quads


def _get_gauss_quad_weights_3():
    """
    Get the weights for the Gaussian quadrature points with 9 points
    """
    # from http://users.metu.edu.tr/csert/me582/ME582%20Ch%2003.pdf
    return np.array([25, 40, 25, 40, 64, 40, 25, 40, 25], dtype=np.float32)/81


def _get_node_weights(q_coords):
    """
    For x, y coordinates in [0, 1] get the weights w_i that each surrounding node contributes to
    evaluating the function at x, y: f(x,y) = w_0 * f_00 + w_1 * f_01 + w_2 * f_10 + w_4 * f_11.
    """
    qx = q_coords[:,0]
    qy = q_coords[:,1]
    wx1 = 1 - qx
    wx2 = qx 
    wy1 = 1 - qy
    wy2 = qy
    return np.vstack([wy1*wx1, wy1*wx2, wy2*wx1, wy2*wx2])


def _get_dx_node_weights(q_coords):
    """
    The weights to evaluate d/dx * f at x, y coordinates, see _get_node_weights
    """
    # derive _get_node_weights wrt x
    qx = q_coords[:,0]
    qy = q_coords[:,1]
    return np.vstack([-(1-qy), 1-qy, -qy, qy])


def _get_dy_node_weights(q_coords):
    """
    The weights to evaluate d/dy * f at x, y coordinates, see _get_node_weights
    """
    # derive _get_node_weights wrt y
    qx = q_coords[:,0]
    qy = q_coords[:,1]
    return np.vstack([-(1-qx), -qx, 1-qx, qx])


@njit
def _value_at_quad_points(im, node_weights):
    """
    Evaluate the value of the image at each quadrature point. The weights that determine each corner's
    contribution to the value at the quadrature point should be calculated a priory with a function
    like _get_node_weights(_get_gauss_quad_points_i()).
    Returned is an array of shape ((im.shape[0]-1)*(im.shape[1]-1), node_weights.shape[1]) where the first
    dimension corresponds to the number of cells and the second to the number of quad points in each cell.
    """
    output = np.empty((im.shape[0]-1, im.shape[1]-1, node_weights.shape[1]), dtype=np.float32)
    for r in range(im.shape[0]-1):
        for c in range(im.shape[1]-1):
            for p in range(node_weights.shape[1]):
                output[r, c, p] = (im[r, c] * node_weights[0, p] +
                                   im[r, c+1] * node_weights[1, p] +
                                   im[r+1, c] * node_weights[2, p] +
                                   im[r+1, c+1] * node_weights[3, p])
    return output.reshape(((im.shape[0]-1)*(im.shape[1]-1), node_weights.shape[1]))


def energy(phi_x, phi_y, im1, im2, node_weights, node_weights_dx, node_weights_dy, quad_weights, L):
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
    f_x = _value_at_quad_points(phi_x, node_weights).ravel()
    f_y = _value_at_quad_points(phi_y, node_weights).ravel()
    g = _value_at_quad_points(im2, node_weights)
    f = _eval_im_at_coords(im1, f_x, f_y, np.mean(im1)).reshape(-1, node_weights.shape[1])
    integrated = np.dot(np.sum((f-g)**2, axis=0), quad_weights)
    # regularisation
    phi_x_dx = _value_at_quad_points(phi_x, node_weights_dx)
    phi_y_dx = _value_at_quad_points(phi_y, node_weights_dx)
    phi_x_dy = _value_at_quad_points(phi_x, node_weights_dy)
    phi_y_dy = _value_at_quad_points(phi_y, node_weights_dy)
    regxx = np.dot(np.sum((phi_x_dx - 1)**2, axis=0), quad_weights)
    regyx = np.dot(np.sum((phi_y_dx)**2, axis=0), quad_weights)
    regxy = np.dot(np.sum((phi_x_dy)**2, axis=0), quad_weights)
    regyy = np.dot(np.sum((phi_y_dy - 1)**2, axis=0), quad_weights)
    return integrated + L * (regxx + regyx + regxy + regyy)
