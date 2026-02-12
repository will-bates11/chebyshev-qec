"""Tests for analysis utilities."""

import numpy as np
import pytest

from chebyshev_qec.analysis import (
    compare_approximation,
    convergence_study,
    find_minimum_degree,
    noise_sweep,
)
from chebyshev_qec.qec_models import BitFlipCode, ShorCode, SteaneCode


class TestConvergenceStudy:
    def test_basic(self):
        result = convergence_study(np.exp, 0, 1, degrees=[4, 8, 12, 16])
        assert len(result.degrees) == 4
        assert len(result.max_errors) == 4
        assert len(result.coefficients) == 4

    def test_error_decreases(self):
        result = convergence_study(np.exp, 0, 1, degrees=[4, 8, 12, 16, 20])
        # Errors should generally decrease with degree
        assert result.max_errors[-1] < result.max_errors[0]

    def test_default_degrees(self):
        result = convergence_study(np.sin, 0, np.pi)
        assert len(result.degrees) == 15  # range(2, 32, 2)

    def test_coefficient_shapes(self):
        result = convergence_study(np.exp, 0, 1, degrees=[5, 10])
        assert result.coefficients[0].shape == (5,)
        assert result.coefficients[1].shape == (10,)


class TestCompareApproximation:
    def test_bitflip(self):
        model = BitFlipCode()
        result = compare_approximation(model, n=15, num_points=100)
        assert result.epsilon.shape == (100,)
        assert result.true_values.shape == (100,)
        assert result.approx_values.shape == (100,)
        assert result.max_error < 1e-8

    def test_steane(self):
        model = SteaneCode()
        result = compare_approximation(model, n=15, num_points=100)
        assert result.max_error < 1e-8

    def test_custom_interval(self):
        model = BitFlipCode()
        result = compare_approximation(model, n=15, a=0.01, b=0.2)
        assert result.epsilon[0] >= 0.01
        assert result.epsilon[-1] <= 0.2


class TestFindMinimumDegree:
    def test_polynomial(self):
        """A degree-3 polynomial should be exact at n=4."""
        f = lambda x: x**3 - 2 * x + 1
        n = find_minimum_degree(f, -1, 1, tol=1e-10, max_n=20)
        assert n <= 4

    def test_exp(self):
        n = find_minimum_degree(np.exp, 0, 1, tol=1e-8, max_n=50)
        assert n <= 15

    def test_returns_max_on_failure(self):
        # Very tight tolerance with very small max_n
        n = find_minimum_degree(np.exp, 0, 10, tol=1e-15, max_n=5)
        assert n == 5


class TestNoiseSweep:
    def test_multiple_models(self):
        models = [BitFlipCode(), SteaneCode(), ShorCode()]
        results = noise_sweep(models, n=15, num_points=50)
        assert len(results) == 3
        for name, result in results.items():
            assert result.epsilon.shape == (50,)
