from __future__ import annotations

import logging
import warnings
from typing import Callable

from tqdm.auto import tqdm

from pymatchseries.utils import ArrayType, DenseArrayType, Matrix, get_dispatcher

logger = logging.getLogger(__name__)


def root_gauss_newton(
    F: Callable[[DenseArrayType], DenseArrayType],
    x0: DenseArrayType,
    DF: Callable[[DenseArrayType], ArrayType],
    max_iterations: int = 50,
    stop_epsilon: float = 0.0,
    start_step: float = 1.0,
    show_progress: bool = False,
) -> DenseArrayType:
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
    total_square_error = f.dot(f)
    logger.info("Initial error {:#.6g}".format(total_square_error))
    step = start_step

    # use tqdm to show a progress bar
    if show_progress:
        iterations = tqdm(range(max_iterations), leave=False)
    else:
        iterations = range(max_iterations)

    dp = get_dispatcher(x0)
    matrix_type = None

    for i in iterations:
        matDF = DF(x)
        if matrix_type is None:
            matrix_type = Matrix.get_matrix_type(matDF)
        dx = matrix_type(matDF).solve_lstsq(f)

        if not dp.all(dp.isfinite(dx)):
            raise RuntimeError("Least squares solving failed.")

        x -= dx
        f = F(x)
        updated_total_square_error = f.dot(f)

        if updated_total_square_error >= total_square_error:
            # If the target functional did not decrease with the update, try to
            # find a smaller step so that it does.
            x += dx
            dx *= -1
            step = _get_stepsize(F=F, x=x, dx=dx, start_step=min(2 * step, 1))
            x += step * dx
            f = F(x)
            updated_total_square_error = f.dot(f)
        else:
            step = 1

        error_difference = total_square_error - updated_total_square_error

        if show_progress:
            iterations.set_description(
                "step_size={:#.2g}, error={:#.5g}, difference={:.1e}".format(
                    step,
                    updated_total_square_error,
                    error_difference,
                )
            )

        if error_difference <= stop_epsilon * updated_total_square_error or dp.isclose(
            updated_total_square_error, 0
        ):
            # convergence is reached
            if show_progress:
                iterations.container.close()
            break

        total_square_error = updated_total_square_error

    else:
        warnings.warn(
            "Reached the maximum number of iterations without reaching stop criterion"
        )

    return x


def _get_stepsize(
    F: Callable[[DenseArrayType], DenseArrayType],
    x: DenseArrayType,
    dx: DenseArrayType,
    start_step: float = 1.0,
    min_step: float = 2**-30,
) -> float:
    """
    Get maximum iteration step width to ensure convergence via geometric search

    Parameters
    ----------
    F
        Function to find root of
    x
        Vector of length N that indicates the current best estimate solution
    dx
        Vector of length N that indicates the delta vector to x
    start_step
        Initial guess for the step
    min_step
        Smallest step to accept

    Returns
    -------
    step
        Largest step that ensures a decrease in energy
    """

    def error_function(v):
        evaluated = F(v)
        return evaluated.dot(evaluated)

    step = start_step

    error = error_function(x)
    updated_error = error_function(x + step * dx)

    while (updated_error >= error) and (step >= min_step):
        step *= 0.5
        updated_error = error_function(x + step * dx)

    return step
