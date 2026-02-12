"""Adaptive code selection using Chebyshev-approximated error probabilities.

Provides utilities for real-time selection of the optimal quantum error-
correcting code given a measured (or estimated) physical error rate, and for
dynamic reconfiguration as noise conditions change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chebyshev_qec.approximation import ChebyshevApproximation
from chebyshev_qec.qec_models import QECModel


@dataclass
class CodeCandidate:
    """A QEC code together with its Chebyshev-approximated error function."""

    model: QECModel
    approximation: ChebyshevApproximation


class AdaptiveCodeSelector:
    """Select the best QEC code for a given noise level in real time.

    For each registered :class:`QECModel`, a :class:`ChebyshevApproximation`
    is precomputed.  At query time, all approximations are evaluated (a fast
    polynomial evaluation) and the code with the lowest estimated logical
    error probability is returned.

    Parameters
    ----------
    models : list of QECModel
        Candidate error-correcting codes.
    n : int
        Default Chebyshev polynomial degree for each code.
    a, b : float or None
        Override the approximation interval.  If *None*, each model's
        ``valid_range`` is used (they must all share the same range).
    """

    def __init__(
        self,
        models: list[QECModel],
        n: int = 20,
        a: float | None = None,
        b: float | None = None,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required")

        self.candidates: list[CodeCandidate] = []
        for model in models:
            lo, hi = model.valid_range
            approx_a = a if a is not None else lo
            approx_b = b if b is not None else hi
            approx = ChebyshevApproximation(
                model.error_probability, approx_a, approx_b, n
            )
            self.candidates.append(CodeCandidate(model=model, approximation=approx))

    def select(self, epsilon: float) -> CodeCandidate:
        """Return the :class:`CodeCandidate` with the lowest estimated P_e.

        Parameters
        ----------
        epsilon : float
            Current physical error rate.

        Returns
        -------
        best : CodeCandidate
            The code whose Chebyshev-approximated error probability is lowest.
        """
        best = min(self.candidates, key=lambda c: c.approximation(epsilon))
        return best

    def rank(self, epsilon: float) -> list[tuple[CodeCandidate, float]]:
        """Rank all candidates by estimated error probability.

        Returns
        -------
        ranked : list of (CodeCandidate, float)
            Candidates sorted by ascending estimated P_e, paired with the
            estimated value.
        """
        scored = [
            (c, float(c.approximation(epsilon))) for c in self.candidates
        ]
        scored.sort(key=lambda pair: pair[1])
        return scored

    def evaluate_all(self, epsilon: ArrayLike) -> dict[str, NDArray[np.float64]]:
        """Evaluate every candidate's approximation over an array of noise levels.

        Returns
        -------
        results : dict mapping code name -> array of estimated P_e values.
        """
        epsilon = np.asarray(epsilon, dtype=np.float64)
        return {
            c.model.name: c.approximation(epsilon) for c in self.candidates
        }


class DynamicReconfigurer:
    """Monitors noise estimates and triggers code reconfiguration.

    Wraps an :class:`AdaptiveCodeSelector` and tracks the currently active
    code.  When a new noise estimate arrives, if a different code would
    perform better by at least *hysteresis*, a switch is recommended.

    Parameters
    ----------
    selector : AdaptiveCodeSelector
        The underlying selector.
    hysteresis : float
        Minimum improvement in estimated P_e required to trigger a switch.
        Prevents rapid oscillation between codes with similar performance.
    """

    def __init__(
        self,
        selector: AdaptiveCodeSelector,
        hysteresis: float = 0.01,
    ) -> None:
        self.selector = selector
        self.hysteresis = hysteresis
        self._current: CodeCandidate | None = None
        self._history: list[tuple[float, str]] = []

    @property
    def current_code(self) -> CodeCandidate | None:
        return self._current

    @property
    def history(self) -> list[tuple[float, str]]:
        """List of (epsilon, code_name) recording every switch."""
        return list(self._history)

    def update(self, epsilon: float) -> CodeCandidate:
        """Process a new noise estimate and return the active code.

        If no code is currently selected, the best code is chosen
        unconditionally.  Otherwise a switch occurs only if the improvement
        exceeds the hysteresis threshold.
        """
        best = self.selector.select(epsilon)

        if self._current is None:
            self._current = best
            self._history.append((epsilon, best.model.name))
            return self._current

        current_pe = float(self._current.approximation(epsilon))
        best_pe = float(best.approximation(epsilon))

        if current_pe - best_pe > self.hysteresis:
            self._current = best
            self._history.append((epsilon, best.model.name))

        return self._current
