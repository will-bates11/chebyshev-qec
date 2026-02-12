"""Quantum error-correcting code models.

Each model provides a ``P_e(epsilon)`` method that computes the logical error
probability as a function of the physical error rate *epsilon*.  These
functions serve as the targets for Chebyshev approximation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import comb

import numpy as np
from numpy.typing import ArrayLike, NDArray


class QECModel(ABC):
    """Abstract base class for quantum error correction models."""

    @abstractmethod
    def error_probability(self, epsilon: ArrayLike) -> NDArray[np.float64]:
        """Compute logical error probability for physical error rate *epsilon*.

        Parameters
        ----------
        epsilon : array_like
            Physical error rate(s) in [0, 1].

        Returns
        -------
        P_e : ndarray
            Logical error probability at each *epsilon*.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the code."""

    @property
    def valid_range(self) -> tuple[float, float]:
        """Default approximation interval for the physical error rate."""
        return (0.0, 0.5)


class BitFlipCode(QECModel):
    """Three-qubit bit-flip repetition code.

    The code corrects a single bit-flip error.  The logical error probability
    is the probability that two or more of the three qubits flip:

        P_e = 3 * epsilon^2 * (1 - epsilon) + epsilon^3
            = 3*epsilon^2 - 2*epsilon^3
    """

    def __init__(self, n_qubits: int = 3) -> None:
        if n_qubits < 3 or n_qubits % 2 == 0:
            raise ValueError("n_qubits must be odd and >= 3")
        self._n = n_qubits

    def error_probability(self, epsilon: ArrayLike) -> NDArray[np.float64]:
        eps = np.asarray(epsilon, dtype=np.float64)
        t = (self._n - 1) // 2  # correctable errors
        # P_e = sum_{k=t+1}^{n} C(n,k) eps^k (1-eps)^{n-k}
        result = np.zeros_like(eps)
        for k in range(t + 1, self._n + 1):
            result += comb(self._n, k) * eps**k * (1 - eps) ** (self._n - k)
        return result

    @property
    def name(self) -> str:
        return f"[{self._n},1,{(self._n - 1) // 2 + 1}] Bit-Flip Code"


class PhaseFlipCode(QECModel):
    """Three-qubit phase-flip code.

    Mathematically identical to the bit-flip code under a Hadamard basis
    change, so the logical error probability formula is the same.
    """

    def __init__(self, n_qubits: int = 3) -> None:
        self._inner = BitFlipCode(n_qubits)

    def error_probability(self, epsilon: ArrayLike) -> NDArray[np.float64]:
        return self._inner.error_probability(epsilon)

    @property
    def name(self) -> str:
        n = self._inner._n
        return f"[{n},1,{(n - 1) // 2 + 1}] Phase-Flip Code"


class ShorCode(QECModel):
    """Shor [[9,1,3]] code.

    Concatenation of a 3-qubit phase-flip code with three 3-qubit bit-flip
    codes.  The logical error rate is modelled as:

        P_e = 1 - (1 - P_bf)^3 * (1 - P_pf)

    where P_bf and P_pf are the logical error probabilities of the component
    bit-flip and phase-flip repetition codes, approximated at the same
    physical error rate.

    A more precise treatment would track X and Z errors separately, but this
    simplified model captures the essential structure.
    """

    def error_probability(self, epsilon: ArrayLike) -> NDArray[np.float64]:
        eps = np.asarray(epsilon, dtype=np.float64)
        p_block = 3 * eps**2 - 2 * eps**3  # single repetition code failure
        # Three bit-flip blocks; outer phase-flip layer
        p_logical = 1 - (1 - p_block) ** 3
        return p_logical

    @property
    def name(self) -> str:
        return "[[9,1,3]] Shor Code"


class SteaneCode(QECModel):
    """Steane [[7,1,3]] code.

    This CSS code corrects any single-qubit error.  Under a depolarising
    noise model the logical error probability is dominated by weight-2 errors:

        P_e ≈ C(7,2) * epsilon^2  (leading order)

    We use the exact binomial sum over uncorrectable error patterns.
    """

    def error_probability(self, epsilon: ArrayLike) -> NDArray[np.float64]:
        eps = np.asarray(epsilon, dtype=np.float64)
        n = 7
        t = 1  # corrects single errors
        result = np.zeros_like(eps)
        for k in range(t + 1, n + 1):
            result += comb(n, k) * eps**k * (1 - eps) ** (n - k)
        return result

    @property
    def name(self) -> str:
        return "[[7,1,3]] Steane Code"


class SurfaceCode(QECModel):
    """Surface code with configurable code distance *d*.

    Under independent depolarising noise, the logical error rate scales as:

        P_e ≈ A * (epsilon / epsilon_th)^{(d+1)/2}

    where epsilon_th ≈ 0.01 is the threshold error rate and A is a
    code-dependent constant (set to 0.1 by default).

    This phenomenological model is widely used in the literature for
    quick estimates.
    """

    def __init__(
        self,
        d: int = 3,
        threshold: float = 0.01,
        prefactor: float = 0.1,
    ) -> None:
        if d < 3 or d % 2 == 0:
            raise ValueError("Code distance d must be odd and >= 3")
        self._d = d
        self._threshold = threshold
        self._prefactor = prefactor

    def error_probability(self, epsilon: ArrayLike) -> NDArray[np.float64]:
        eps = np.asarray(epsilon, dtype=np.float64)
        exponent = (self._d + 1) / 2
        p_logical = self._prefactor * (eps / self._threshold) ** exponent
        return np.clip(p_logical, 0.0, 1.0)

    @property
    def name(self) -> str:
        return f"[[d={self._d}]] Surface Code"

    @property
    def valid_range(self) -> tuple[float, float]:
        return (0.0, min(0.5, 2 * self._threshold))


class DepolarizingCode(QECModel):
    """Generic [[n, k, d]] code under depolarising noise.

    Uses the exact binomial model: the code fails when more than t = (d-1)/2
    errors occur among *n* qubits, each independently depolarised with
    probability *epsilon*.
    """

    def __init__(self, n: int, k: int, d: int) -> None:
        if d < 1 or d % 2 == 0:
            raise ValueError("Distance d must be odd and >= 1")
        self._n = n
        self._k = k
        self._d = d
        self._t = (d - 1) // 2

    def error_probability(self, epsilon: ArrayLike) -> NDArray[np.float64]:
        eps = np.asarray(epsilon, dtype=np.float64)
        result = np.zeros_like(eps)
        for k in range(self._t + 1, self._n + 1):
            result += comb(self._n, k) * eps**k * (1 - eps) ** (self._n - k)
        return result

    @property
    def name(self) -> str:
        return f"[[{self._n},{self._k},{self._d}]] Depolarizing Code"
