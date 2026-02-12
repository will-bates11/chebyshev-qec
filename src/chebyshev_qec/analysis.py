"""Analysis and simulation utilities.

Helper functions for convergence studies, error landscapes, and comparative
benchmarks of Chebyshev-approximated QEC error probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from chebyshev_qec.approximation import ChebyshevApproximation
from chebyshev_qec.qec_models import QECModel


@dataclass
class ConvergenceResult:
    """Result of a convergence study over polynomial degrees."""

    degrees: list[int]
    max_errors: list[float]
    coefficients: list[NDArray[np.float64]] = field(repr=False)


def convergence_study(
    f: callable,
    a: float,
    b: float,
    degrees: list[int] | None = None,
    num_test_points: int = 1000,
) -> ConvergenceResult:
    """Study how approximation error decreases with polynomial degree.

    Parameters
    ----------
    f : callable
        Target function.
    a, b : float
        Approximation interval.
    degrees : list of int, optional
        Polynomial degrees to test.  Defaults to [2, 4, 6, ..., 30].
    num_test_points : int
        Number of equally spaced test points for error estimation.

    Returns
    -------
    result : ConvergenceResult
    """
    if degrees is None:
        degrees = list(range(2, 32, 2))

    max_errors: list[float] = []
    all_coeffs: list[NDArray[np.float64]] = []

    for n in degrees:
        approx = ChebyshevApproximation(f, a, b, n)
        max_errors.append(approx.max_error(num_test_points))
        all_coeffs.append(approx.coeffs)

    return ConvergenceResult(
        degrees=degrees, max_errors=max_errors, coefficients=all_coeffs
    )


@dataclass
class ComparisonResult:
    """Comparison of approximation vs true error probability for a QEC model."""

    epsilon: NDArray[np.float64]
    true_values: NDArray[np.float64]
    approx_values: NDArray[np.float64]
    pointwise_error: NDArray[np.float64]
    max_error: float
    mean_error: float


def compare_approximation(
    model: QECModel,
    n: int = 20,
    a: float | None = None,
    b: float | None = None,
    num_points: int = 500,
) -> ComparisonResult:
    """Compare exact and Chebyshev-approximated error probabilities.

    Parameters
    ----------
    model : QECModel
        The quantum error correction model.
    n : int
        Chebyshev polynomial degree.
    a, b : float or None
        Approximation interval (defaults to model.valid_range).
    num_points : int
        Number of evaluation points.

    Returns
    -------
    result : ComparisonResult
    """
    lo, hi = model.valid_range
    a = a if a is not None else lo
    b = b if b is not None else hi

    approx = ChebyshevApproximation(model.error_probability, a, b, n)
    eps = np.linspace(a, b, num_points)
    true_vals = np.asarray(model.error_probability(eps), dtype=np.float64)
    approx_vals = approx(eps)
    pw_err = np.abs(true_vals - approx_vals)

    return ComparisonResult(
        epsilon=eps,
        true_values=true_vals,
        approx_values=approx_vals,
        pointwise_error=pw_err,
        max_error=float(np.max(pw_err)),
        mean_error=float(np.mean(pw_err)),
    )


def find_minimum_degree(
    f: callable,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_n: int = 100,
    num_test_points: int = 1000,
) -> int:
    """Find the minimum Chebyshev degree to achieve a target accuracy.

    Parameters
    ----------
    f : callable
        Target function.
    a, b : float
        Approximation interval.
    tol : float
        Target maximum absolute error.
    max_n : int
        Upper bound on degree to search.
    num_test_points : int
        Test grid density.

    Returns
    -------
    n : int
        Smallest degree achieving the tolerance, or *max_n* if not reached.
    """
    for n in range(1, max_n + 1):
        approx = ChebyshevApproximation(f, a, b, n)
        if approx.max_error(num_test_points) < tol:
            return n
    return max_n


def noise_sweep(
    models: list[QECModel],
    n: int = 20,
    num_points: int = 500,
) -> dict[str, ComparisonResult]:
    """Run :func:`compare_approximation` for multiple QEC models.

    Returns
    -------
    results : dict mapping model name -> ComparisonResult
    """
    return {
        model.name: compare_approximation(model, n=n, num_points=num_points)
        for model in models
    }
