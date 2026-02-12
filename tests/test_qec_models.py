"""Tests for quantum error correction models."""

import numpy as np
import pytest

from chebyshev_qec.qec_models import (
    BitFlipCode,
    DepolarizingCode,
    PhaseFlipCode,
    ShorCode,
    SteaneCode,
    SurfaceCode,
)


class TestBitFlipCode:
    def test_zero_noise(self):
        code = BitFlipCode()
        assert np.isclose(code.error_probability(0.0), 0.0)

    def test_max_noise(self):
        code = BitFlipCode()
        assert np.isclose(code.error_probability(0.5), 0.5)

    def test_known_value_n3(self):
        """P_e = 3*eps^2 - 2*eps^3 for 3-qubit code."""
        code = BitFlipCode(3)
        eps = 0.1
        expected = 3 * eps**2 - 2 * eps**3
        assert np.isclose(code.error_probability(eps), expected, atol=1e-14)

    def test_array_input(self):
        code = BitFlipCode()
        eps = np.array([0.0, 0.1, 0.2, 0.5])
        result = code.error_probability(eps)
        assert result.shape == (4,)
        assert np.all(result >= 0)

    def test_5_qubit(self):
        code = BitFlipCode(5)
        # With 5 qubits, corrects up to 2 errors
        assert np.isclose(code.error_probability(0.0), 0.0)
        assert code.error_probability(0.1) < BitFlipCode(3).error_probability(0.1)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            BitFlipCode(2)
        with pytest.raises(ValueError):
            BitFlipCode(4)

    def test_name(self):
        assert "Bit-Flip" in BitFlipCode().name


class TestPhaseFlipCode:
    def test_matches_bit_flip(self):
        """Phase-flip code has the same error probability as bit-flip."""
        bf = BitFlipCode()
        pf = PhaseFlipCode()
        eps = np.linspace(0, 0.5, 50)
        assert np.allclose(bf.error_probability(eps), pf.error_probability(eps))

    def test_name(self):
        assert "Phase-Flip" in PhaseFlipCode().name


class TestShorCode:
    def test_zero_noise(self):
        code = ShorCode()
        assert np.isclose(code.error_probability(0.0), 0.0)

    def test_monotonic(self):
        code = ShorCode()
        eps = np.linspace(0, 0.3, 100)
        pe = code.error_probability(eps)
        assert np.all(np.diff(pe) >= -1e-15)

    def test_name(self):
        assert "Shor" in ShorCode().name


class TestSteaneCode:
    def test_zero_noise(self):
        code = SteaneCode()
        assert np.isclose(code.error_probability(0.0), 0.0)

    def test_low_noise_quadratic(self):
        """At low noise, P_e ≈ 21*eps^2 (leading order)."""
        code = SteaneCode()
        eps = 1e-4
        pe = code.error_probability(eps)
        # Leading order: C(7,2)*eps^2 = 21*eps^2
        assert np.isclose(pe, 21 * eps**2, rtol=0.01)

    def test_name(self):
        assert "Steane" in SteaneCode().name


class TestSurfaceCode:
    def test_zero_noise(self):
        code = SurfaceCode(d=3)
        assert np.isclose(code.error_probability(0.0), 0.0)

    def test_clipping(self):
        code = SurfaceCode(d=3)
        pe = code.error_probability(0.5)
        assert pe <= 1.0

    def test_higher_distance_improves(self):
        eps = 0.005  # Below threshold
        pe_3 = SurfaceCode(d=3).error_probability(eps)
        pe_5 = SurfaceCode(d=5).error_probability(eps)
        pe_7 = SurfaceCode(d=7).error_probability(eps)
        assert pe_5 < pe_3
        assert pe_7 < pe_5

    def test_invalid_distance(self):
        with pytest.raises(ValueError):
            SurfaceCode(d=2)
        with pytest.raises(ValueError):
            SurfaceCode(d=4)

    def test_name(self):
        assert "Surface" in SurfaceCode(d=5).name


class TestDepolarizingCode:
    def test_matches_steane(self):
        """DepolarizingCode(7,1,3) should match Steane code."""
        dc = DepolarizingCode(7, 1, 3)
        sc = SteaneCode()
        eps = np.linspace(0, 0.3, 50)
        assert np.allclose(dc.error_probability(eps), sc.error_probability(eps))

    def test_zero_noise(self):
        code = DepolarizingCode(5, 1, 3)
        assert np.isclose(code.error_probability(0.0), 0.0)

    def test_name(self):
        assert "Depolarizing" in DepolarizingCode(5, 1, 3).name
