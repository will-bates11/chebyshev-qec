"""Core Chebyshev approximation machinery.

Provides routines for computing Chebyshev nodes, coefficients, and evaluating
Chebyshev polynomial approximations of functions defined on arbitrary
intervals [a, b].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def chebyshev_nodes(a: float, b: float, n: int) -> NDArray[np.float64]:
    """Compute *n* Chebyshev nodes mapped to the interval [a, b].

    The nodes are the roots of the degree-*n* Chebyshev polynomial of the
    first kind, linearly transformed from [-1, 1] to [a, b]:

        x_k = (a+b)/2 + (b-a)/2 * cos((2k-1)*pi / (2n)),  k = 1, ..., n

    Parameters
    ----------
    a, b : float
        Endpoints of the approximation interval (a < b).
    n : int
        Number of nodes (must be >= 1).

    Returns
    -------
    nodes : ndarray of shape (n,)
        Chebyshev nodes in descending order within [a, b].
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if a >= b:
        raise ValueError("Require a < b")
    k = np.arange(1, n + 1)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))


def chebyshev_coefficients(
    f: callable,
    a: float,
    b: float,
    n: int,
) -> NDArray[np.float64]:
    """Compute the first *n* Chebyshev expansion coefficients of *f* on [a, b].

    Uses discrete orthogonality at the Chebyshev nodes to compute

        c_j = (2/n) * sum_{k=1}^{n} f(x_k) * cos(j*(2k-1)*pi / (2n))

    with c_0 halved so the reconstruction formula is simply sum_j c_j T_j(x').

    Parameters
    ----------
    f : callable
        The function to approximate.  Must accept an ndarray and return an
        ndarray of the same shape.
    a, b : float
        Interval endpoints.
    n : int
        Number of terms (and nodes) to use.

    Returns
    -------
    coeffs : ndarray of shape (n,)
        Chebyshev coefficients c_0, c_1, ..., c_{n-1}.  Note that c_0 is
        already halved for direct use in evaluation.
    """
    nodes = chebyshev_nodes(a, b, n)
    y = np.asarray(f(nodes), dtype=np.float64)
    k = np.arange(1, n + 1)
    coeffs = np.zeros(n, dtype=np.float64)
    for j in range(n):
        coeffs[j] = (2.0 / n) * np.sum(
            y * np.cos(j * (2 * k - 1) * np.pi / (2 * n))
        )
    # Halve c_0 so the evaluation formula is simply dot(c, T)
    coeffs[0] /= 2.0
    return coeffs


def _evaluate_chebyshev(
    coeffs: NDArray[np.float64],
    a: float,
    b: float,
    x: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluate a Chebyshev expansion at arbitrary points.

    Uses the recurrence relation T_{j+1}(t) = 2t T_j(t) - T_{j-1}(t) for
    numerical stability (Clenshaw's algorithm could be used as well, but the
    direct form is clearer here).

    Parameters
    ----------
    coeffs : ndarray
        Chebyshev coefficients (c_0 already halved).
    a, b : float
        Original interval.
    x : array_like
        Points at which to evaluate.

    Returns
    -------
    result : ndarray
        Approximation values at *x*.
    """
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    # Map to [-1, 1]
    x_prime = (2.0 * x - (b + a)) / (b - a)

    n = len(coeffs)
    if n == 0:
        result = np.zeros_like(x)
    elif n == 1:
        result = np.full_like(x, coeffs[0])
    else:
        # Use Clenshaw recurrence for stability
        b_prev = np.zeros_like(x)  # b_{n}
        b_curr = np.zeros_like(x)  # b_{n-1}
        for j in range(n - 1, 0, -1):
            b_next = coeffs[j] + 2.0 * x_prime * b_curr - b_prev
            b_prev = b_curr
            b_curr = b_next
        result = coeffs[0] + x_prime * b_curr - b_prev

    return float(result[0]) if scalar_input else result


class ChebyshevApproximation:
    """A Chebyshev polynomial approximation of a function on [a, b].

    Parameters
    ----------
    f : callable
        Target function mapping array -> array.
    a, b : float
        Approximation interval.
    n : int
        Polynomial degree (number of terms).

    Examples
    --------
    >>> import numpy as np
    >>> approx = ChebyshevApproximation(np.exp, 0.0, 1.0, 15)
    >>> abs(approx(0.5) - np.exp(0.5)) < 1e-12
    True
    """

    def __init__(self, f: callable, a: float, b: float, n: int) -> None:
        self.f = f
        self.a = a
        self.b = b
        self.n = n
        self.coeffs = chebyshev_coefficients(f, a, b, n)

    def __call__(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the approximation at *x*."""
        return _evaluate_chebyshev(self.coeffs, self.a, self.b, x)

    def max_error(self, num_test_points: int = 1000) -> float:
        """Estimate the maximum approximation error on [a, b].

        Evaluates both the true function and the approximation at
        *num_test_points* equally spaced points and returns the largest
        absolute difference.
        """
        xs = np.linspace(self.a, self.b, num_test_points)
        true_vals = np.asarray(self.f(xs), dtype=np.float64)
        approx_vals = self(xs)
        return float(np.max(np.abs(true_vals - approx_vals)))

    def refine(self, new_n: int) -> "ChebyshevApproximation":
        """Return a new approximation with a different polynomial degree."""
        return ChebyshevApproximation(self.f, self.a, self.b, new_n)

    def adaptive_refine(
        self,
        tol: float = 1e-8,
        max_n: int = 100,
        num_test_points: int = 1000,
    ) -> "ChebyshevApproximation":
        """Iteratively increase degree until *tol* is met or *max_n* reached.

        Returns
        -------
        approx : ChebyshevApproximation
            A refined approximation meeting the tolerance, or the best
            approximation found up to *max_n* terms.
        """
        current = self
        for n in range(current.n + 1, max_n + 1):
            current = current.refine(n)
            if current.max_error(num_test_points) < tol:
                break
        return current
