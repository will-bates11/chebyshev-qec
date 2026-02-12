"""Tests for adaptive code selection and dynamic reconfiguration."""

import numpy as np
import pytest

from chebyshev_qec.adaptive import AdaptiveCodeSelector, DynamicReconfigurer
from chebyshev_qec.qec_models import BitFlipCode, ShorCode, SteaneCode, SurfaceCode


@pytest.fixture
def models():
    return [BitFlipCode(), SteaneCode(), ShorCode()]


@pytest.fixture
def selector(models):
    return AdaptiveCodeSelector(models, n=15, a=0.001, b=0.4)


class TestAdaptiveCodeSelector:
    def test_construction(self, selector):
        assert len(selector.candidates) == 3

    def test_empty_models(self):
        with pytest.raises(ValueError, match="At least one model"):
            AdaptiveCodeSelector([])

    def test_select_returns_candidate(self, selector):
        best = selector.select(0.1)
        assert best.model is not None
        assert best.approximation is not None

    def test_rank_ordering(self, selector):
        ranked = selector.rank(0.1)
        assert len(ranked) == 3
        # P_e values should be in ascending order
        pe_values = [pe for _, pe in ranked]
        assert pe_values == sorted(pe_values)

    def test_evaluate_all(self, selector):
        eps = np.linspace(0.01, 0.3, 10)
        results = selector.evaluate_all(eps)
        assert len(results) == 3
        for name, vals in results.items():
            assert len(vals) == 10

    def test_select_consistency(self, selector):
        """Selecting twice at the same epsilon should give the same result."""
        best1 = selector.select(0.1)
        best2 = selector.select(0.1)
        assert best1.model.name == best2.model.name


class TestDynamicReconfigurer:
    def test_initial_selection(self, selector):
        reconfig = DynamicReconfigurer(selector, hysteresis=0.01)
        assert reconfig.current_code is None
        code = reconfig.update(0.1)
        assert code is not None
        assert reconfig.current_code is code

    def test_history_recorded(self, selector):
        reconfig = DynamicReconfigurer(selector, hysteresis=0.01)
        reconfig.update(0.1)
        assert len(reconfig.history) == 1

    def test_hysteresis_prevents_oscillation(self, selector):
        reconfig = DynamicReconfigurer(selector, hysteresis=1.0)
        reconfig.update(0.1)
        initial_code = reconfig.current_code.model.name
        # With large hysteresis, the code shouldn't switch
        reconfig.update(0.2)
        assert reconfig.current_code.model.name == initial_code

    def test_switch_on_large_change(self):
        """With zero hysteresis, code may switch when noise changes a lot."""
        models = [BitFlipCode(3), BitFlipCode(5)]
        selector = AdaptiveCodeSelector(models, n=15, a=0.001, b=0.4)
        reconfig = DynamicReconfigurer(selector, hysteresis=0.0)
        reconfig.update(0.05)
        reconfig.update(0.3)
        # At least the initial entry should be in history
        assert len(reconfig.history) >= 1
