import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_strategy import V2_PROFILE, build_cost_model
from portfolio_profiles import (
    PortfolioProfile,
    SIX_COIN_V2,
    THREE_COIN_V2,
    build_strategies,
    get_portfolio_profile,
)


class PortfolioProfileTests(unittest.TestCase):
    def test_six_coin_profile_matches_researched_settings(self):
        expected = {
            "ADA-USD": (24, 0.70),
            "DOGE-USD": (48, 0.68),
            "LINK-USD": (72, 0.70),
            "SOL-USD": (72, 0.65),
            "XLM-USD": (72, 0.70),
            "XRP-USD": (72, 0.65),
        }
        self.assertEqual(set(SIX_COIN_V2.symbols), set(expected))
        for symbol, (horizon, threshold) in expected.items():
            with self.subTest(symbol=symbol):
                strategy = SIX_COIN_V2.symbols[symbol]
                self.assertEqual(strategy.horizon, horizon)
                self.assertEqual(strategy.buy_thr, threshold)
                self.assertEqual(strategy.exit_thr, 0.40)
                self.assertTrue(strategy.long_only)
                self.assertTrue(strategy.use_fng_features)
                self.assertTrue(strategy.use_fng_filter)
                self.assertFalse(strategy.use_cost_aware_labels)
                self.assertFalse(strategy.use_ev_exit)
        self.assertEqual(SIX_COIN_V2.position_fraction, 0.33)
        self.assertEqual(SIX_COIN_V2.leverage, 2)
        self.assertEqual(SIX_COIN_V2.max_open, 3)
        self.assertTrue(SIX_COIN_V2.use_dynamic_threshold)

    def test_baseline_matches_current_three_coin_live_settings(self):
        self.assertEqual(set(THREE_COIN_V2.symbols), {"ADA-USD", "DOGE-USD", "SOL-USD"})
        self.assertFalse(THREE_COIN_V2.use_dynamic_threshold)
        for strategy in THREE_COIN_V2.symbols.values():
            self.assertEqual((strategy.horizon, strategy.buy_thr, strategy.exit_thr),
                             (24, 0.70, 0.40))

    def test_strategies_use_per_coin_settings_and_are_independent(self):
        costs = build_cost_model(V2_PROFILE)
        strategies = build_strategies(SIX_COIN_V2, costs)
        second_run = build_strategies(SIX_COIN_V2, costs)
        self.assertEqual(len({id(strategy) for strategy in strategies.values()}), 6)
        for symbol, strategy in strategies.items():
            with self.subTest(symbol=symbol):
                saved = SIX_COIN_V2.symbols[symbol]
                self.assertEqual(strategy.horizon, saved.horizon)
                self.assertEqual(strategy.buy_thr, saved.buy_thr)
                self.assertEqual(strategy.exit_thr, saved.exit_thr)
                self.assertTrue(strategy.use_dynamic_threshold)
                self.assertIs(strategy.cost_model, costs)
                self.assertIsNot(strategy, second_run[symbol])
        strategies["SOL-USD"].buy_thr = 0.99
        self.assertEqual(strategies["ADA-USD"].buy_thr, 0.70)
        self.assertEqual(second_run["SOL-USD"].buy_thr, 0.65)
        self.assertEqual(SIX_COIN_V2.symbols["SOL-USD"].buy_thr, 0.65)

    def test_baseline_strategies_preserve_fixed_live_thresholds(self):
        strategies = build_strategies(THREE_COIN_V2, build_cost_model(V2_PROFILE))
        for symbol, strategy in strategies.items():
            with self.subTest(symbol=symbol):
                self.assertFalse(strategy.use_dynamic_threshold)
                self.assertEqual(strategy.buy_thr, 0.70)

    def test_candidate_cost_floor_can_block_signal_that_fixed_baseline_accepts(self):
        costs = build_cost_model(V2_PROFILE)
        # ADA's nominal horizon/threshold are the same in both portfolios.
        candidate = build_strategies(SIX_COIN_V2, costs)["ADA-USD"]
        baseline = build_strategies(THREE_COIN_V2, costs)["ADA-USD"]
        data = pd.DataFrame({"Close": [100, 101, 100]},
                            index=pd.date_range("2026-01-01", periods=3, freq="h"))
        prediction = (0.71, None, pd.Series([0.01, -0.01]), np.array([True, True]))
        with patch.object(candidate, "_train_and_predict", return_value=prediction), \
                patch.object(baseline, "_train_and_predict", return_value=prediction), \
                patch("ml_strategy.passes_fng_filter", return_value=True):
            candidate_signal = candidate.get_signal(data)
            baseline_signal = baseline.get_signal(data)
        self.assertIsNone(candidate_signal.signal)
        self.assertGreater(candidate_signal.dynamic_threshold, 0.71)
        self.assertEqual(baseline_signal.signal, "BUY")
        self.assertEqual(baseline_signal.dynamic_threshold, 0.70)

    def test_profile_and_symbol_map_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            SIX_COIN_V2.max_open = 99
        with self.assertRaises(TypeError):
            SIX_COIN_V2.symbols["BTC-USD"] = V2_PROFILE
        with self.assertRaises(FrozenInstanceError):
            SIX_COIN_V2.symbols["ADA-USD"].buy_thr = 0.1
        original = {"ADA-USD": V2_PROFILE}
        custom = PortfolioProfile("custom", original)
        original.clear()
        self.assertEqual(list(custom.symbols), ["ADA-USD"])

    def test_lookup_is_explicit_and_rejects_invalid_names(self):
        self.assertIs(get_portfolio_profile("six_coin_v2"), SIX_COIN_V2)
        self.assertIs(get_portfolio_profile(" THREE_COIN_V2 "), THREE_COIN_V2)
        for name in ("", "v2", "six_coin_v3", None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown portfolio"):
                get_portfolio_profile(name)


if __name__ == "__main__":
    unittest.main()
