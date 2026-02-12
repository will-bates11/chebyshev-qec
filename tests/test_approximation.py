"""Tests for the Chebyshev approximation core module."""

import numpy as np
import pytest

from chebyshev_qec.approximation import (
    ChebyshevApproximation,
    chebyshev_coefficients,
    chebyshev_nodes,
)


class TestChebyshevNodes:
    def test_count(self):
        nodes = chebyshev_nodes(0, 1, 10)
        assert len(nodes) == 10

    def test_within_interval(self):
        a, b = -2.0, 3.0
        nodes = chebyshev_nodes(a, b, 20)
        assert np.all(nodes >= a)
        assert np.all(nodes <= b)

    def test_symmetry_on_symmetric_interval(self):
        nodes = chebyshev_nodes(-1, 1, 8)
        # Nodes should be symmetric about 0: sorted nodes + reversed sorted nodes = 0
        sorted_nodes = np.sort(nodes)
        assert np.allclose(sorted_nodes, -sorted_nodes[::-1])

    def test_n_equals_1(self):
        nodes = chebyshev_nodes(0, 1, 1)
        assert len(nodes) == 1
        # Single node should be at the midpoint cos(pi/2) mapped to [0,1]
        expected = 0.5 + 0.5 * np.cos(np.pi / 2)
        assert np.isclose(nodes[0], expected)

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            chebyshev_nodes(0, 1, 0)

    def test_invalid_interval(self):
        with pytest.raises(ValueError, match="Require a < b"):
            chebyshev_nodes(1, 0, 5)
        with pytest.raises(ValueError, match="Require a < b"):
            chebyshev_nodes(1, 1, 5)


class TestChebyshevCoefficients:
    def test_constant_function(self):
        """Constant function should have c_0 = constant, rest zero."""
        f = lambda x: np.full_like(x, 3.0)
        coeffs = chebyshev_coefficients(f, 0, 1, 10)
        assert np.isclose(coeffs[0], 3.0, atol=1e-12)
        assert np.allclose(coeffs[1:], 0, atol=1e-12)

    def test_linear_function(self):
        """Linear function x on [0,1]: c_0 = 0.5, c_1 = 0.5, rest ≈ 0."""
        f = lambda x: x
        coeffs = chebyshev_coefficients(f, 0, 1, 10)
        # Reconstruct at midpoint: should give 0.5
        approx = ChebyshevApproximation(f, 0, 1, 10)
        assert np.isclose(approx(0.5), 0.5, atol=1e-12)

    def test_coefficient_count(self):
        coeffs = chebyshev_coefficients(np.sin, 0, np.pi, 15)
        assert len(coeffs) == 15


class TestChebyshevApproximation:
    def test_exp_accuracy(self):
        """exp(x) on [0, 1] should be well-approximated with n=15."""
        approx = ChebyshevApproximation(np.exp, 0.0, 1.0, 15)
        xs = np.linspace(0, 1, 50)
        assert np.allclose(approx(xs), np.exp(xs), atol=1e-12)

    def test_sin_accuracy(self):
        """sin(x) on [0, pi] should be well-approximated with n=15."""
        approx = ChebyshevApproximation(np.sin, 0.0, np.pi, 15)
        xs = np.linspace(0, np.pi, 50)
        assert np.allclose(approx(xs), np.sin(xs), atol=1e-12)

    def test_polynomial_exact(self):
        """A degree-3 polynomial should be represented exactly with n >= 4."""
        f = lambda x: 2 * x**3 - x**2 + 0.5 * x - 1
        approx = ChebyshevApproximation(f, -1.0, 1.0, 4)
        xs = np.linspace(-1, 1, 100)
        assert np.allclose(approx(xs), f(xs), atol=1e-10)

    def test_scalar_input(self):
        approx = ChebyshevApproximation(np.exp, 0.0, 1.0, 10)
        result = approx(0.5)
        assert isinstance(result, float)
        assert np.isclose(result, np.exp(0.5), atol=1e-10)

    def test_max_error(self):
        approx = ChebyshevApproximation(np.exp, 0.0, 1.0, 10)
        err = approx.max_error(500)
        assert err < 1e-10

    def test_refine(self):
        approx_low = ChebyshevApproximation(np.exp, 0.0, 1.0, 5)
        approx_high = approx_low.refine(15)
        assert approx_high.n == 15
        assert approx_high.max_error() < approx_low.max_error()

    def test_adaptive_refine(self):
        approx = ChebyshevApproximation(np.exp, 0.0, 1.0, 3)
        refined = approx.adaptive_refine(tol=1e-10, max_n=30)
        assert refined.max_error() < 1e-10
        assert refined.n <= 30

    def test_narrow_interval(self):
        approx = ChebyshevApproximation(np.exp, 0.0, 0.01, 5)
        xs = np.linspace(0, 0.01, 20)
        assert np.allclose(approx(xs), np.exp(xs), atol=1e-12)
