from typing import Callable
import warnings
import logging
from tqdm import tqdm

import numpy as np
from sksparse.cholmod import cholesky_AAt


logger = logging.getLogger(__name__)


def _get_timestep_width_line_search(
    E: Callable,
    dx: np.ndarray,
    x: np.ndarray,
    start_step: float = 1.,
    min_step: float = 2**-30,
) -> float:
    """
    Get iteration step width function to ensure convergence

    Parameters
    ----------
    E
        Error function, taking the direction vector as argument
    dx
        Vector of length N that indicates the delta to x
    x
        Vector of length N that indicates the current best estimate solution
    start_step
        Initial guess for the step
    min_step
        Smallest step to accept

    Returns
    -------
    step
        Step that ensures a decrease in energy
    """
    step = start_step

    error = E(x)
    next_error = E(x + step * dx)

    while (next_error >= error) and (step >= min_step):
        step *= 0.5
        next_error = E(x + step * dx)

    return step


def root_gauss_newton(
    F: Callable,
    x0: np.ndarray,
    DF: Callable,
    max_iterations: int = 50,
    stop_epsilon: float = 0.,
    start_step: float = 1.,
):
    """
    Implementation of Gauss-Newton iterative solver for sparse systems

    Parameters
    ----------
    F
        Function to find the root of
    x0
        Initial guess
    DF
        Function that returns the Jacobian of F
    max_iterations
        Maximum number of iterations
    stop_epsilon
        Relative error change in a step at which to stop iteration
    start_step
        Initial step size

    Returns
    -------
    x
        Root of F
    """
    x = x0.copy()
    f = F(x)
    total_square_error = np.dot(f, f)
    logger.info("Initial error {:#.6g}".format(total_square_error))
    step = start_step

    # use tqdm to show a progress bar
    i_bar = tqdm(range(max_iterations))

    def evaluate_error(v):
        evaluated = F(v)
        return np.dot(evaluated, evaluated)

    for i in i_bar:
        matDF = DF(x)
        # Solve the linear least-squares sense using a Cholesky factorization
        # of the normal equations. Note it would better to directly assemble
        # the transposed instead of assembling and then transposing.
        A = matDF.T
        factor = cholesky_AAt(A)
        dx = factor.solve_A(A * f)

        if not np.all(np.isfinite(dx)):
            raise RuntimeError("Least squares solving failed, .")

        x -= dx

        f = F(x)
        updated_total_square_error = np.dot(f, f)

        # If the target functional did not decrease with the update, try to
        # find a smaller step so that it does. This step size control is
        # extremely simple and not very efficient, but it's certainly better
        # than letting the algorithm diverge.
        if updated_total_square_error >= total_square_error:
            x += dx
            dx *= -1
            # getTimestepWidthWithSimpleLineSearch doesn't support "widening",
            # so let it start with 2*step.
            step = _get_timestep_width_line_search(
                evaluate_error,
                dx,
                x,
                start_step=min(2 * step, 1),
            )
            x += step * dx
            f = F(x)
            updated_total_square_error = np.dot(f, f)
        else:
            step = 1

        i_bar.set_description(
            "\u03C4={:#.2g}, E={:#.5g}, \u0394={:.1e}".format(
                step,
                updated_total_square_error,
                total_square_error - updated_total_square_error,
            )
        )

        if ((total_square_error - updated_total_square_error)) <= stop_epsilon * updated_total_square_error or np.isclose(
            updated_total_square_error, 0
        ):
            break
        total_square_error = updated_total_square_error
    else:
        warnings.warn("Reached the maximum number of iterations without reaching stop critereon")

    return x
