"""Additional allocation, rollover, and time-boundary checks for offline replay."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import pandas as pd

from portfolio_profiles import THREE_COIN_V2
from research_six_coin_validation import (
    HistoricalCosts,
    allocate_candidates,
    mark_to_market,
    regenerate_candidates,
)


ZERO_COST = HistoricalCosts("zero", 0, 0, 0, 0, 0)


def candidate(symbol="ADA-USD", start="2024-01-01 00:00", end="2024-01-01 02:00", score=1, exit_price=100):
    entry, exit_time = pd.Timestamp(start), pd.Timestamp(end)
    return {
        "symbol": symbol, "entry_time": entry, "exit_time": exit_time,
        "entry_price": 100.0, "exit_price": float(exit_price),
        "score": score, "prob_up": 0.8, "direction": "long", "exit_reason": "time",
        "bars_held": int((exit_time - entry).total_seconds() / 3600),
    }


def history(closes, lows=None):
    return pd.DataFrame(
        {"Close": closes, "Low": closes if lows is None else lows},
        index=pd.date_range("2024-01-01", periods=len(closes), freq="h"),
    )


class ValidationAccountingTests(unittest.TestCase):
    def test_cap_and_score_order_reserve_margin_until_exit(self):
        profile = replace(THREE_COIN_V2, max_open=1, position_fraction=0.5)
        entries = pd.DataFrame([
            candidate(score=1), candidate(symbol="DOGE-USD", score=2, exit_price=110),
            candidate(start="2024-01-01 02:00", end="2024-01-01 04:00", exit_price=110),
        ])
        selected, stats = allocate_candidates(entries, profile, ZERO_COST, 100)
        self.assertEqual(list(selected.symbol), ["DOGE-USD", "ADA-USD"])
        self.assertAlmostEqual(selected.margin.iloc[0], 50.0)
        self.assertAlmostEqual(selected.margin.iloc[1], 55.0)
        self.assertEqual(stats["rejected_position_cap"], 1)
        self.assertAlmostEqual(stats["ending_equity"], 121.0)
        self.assertEqual(stats["max_open"], 1)

    def test_same_coin_cannot_occupy_two_slots(self):
        entries = pd.DataFrame([candidate(score=2), candidate(score=1)])
        selected, stats = allocate_candidates(entries, THREE_COIN_V2, ZERO_COST, 100)
        self.assertEqual(len(selected), 1)
        self.assertEqual(stats["rejected_occupied_symbol"], 1)

    def test_mtm_reveals_loss_hidden_by_realized_equity(self):
        profile = replace(THREE_COIN_V2, position_fraction=0.5)
        selected, stats = allocate_candidates(pd.DataFrame([candidate(exit_price=110)]), profile, ZERO_COST, 100)
        curve, risk = mark_to_market(selected, {"ADA-USD": history([100, 70, 110], [1, 60, 90])}, ZERO_COST, 2, 100)
        self.assertEqual(stats["realized_drawdown_pct"], 0)
        self.assertAlmostEqual(risk["close_mtm_drawdown_pct"], 30)
        self.assertAlmostEqual(risk["adverse_hour_drawdown_pct"], 40)
        self.assertEqual(curve.adverse_hour_equity.iloc[0], 100)  # exclude pre-entry candle low
        self.assertEqual(curve.close_mtm_equity.iloc[-1], stats["ending_equity"])
        self.assertAlmostEqual(stats["ending_equity"], 110)

    def test_margin_reservations_cannot_exceed_equity(self):
        entries = pd.DataFrame([candidate(symbol=symbol, score=3-i) for i, symbol in enumerate(THREE_COIN_V2.symbols)])
        selected, stats = allocate_candidates(entries, replace(THREE_COIN_V2, position_fraction=0.7), ZERO_COST, 100)
        self.assertEqual(list(selected.margin), [70.0, 30.0])
        self.assertEqual(stats["max_margin_fraction"], 1)
        self.assertEqual(stats["rejected_no_margin"], 1)

    def test_fee_reserve_not_double_counted_at_exit(self):
        profile = replace(THREE_COIN_V2, position_fraction=0.5)
        costs = HistoricalCosts("fees", 0.01, 0.01, 0, 0, 0)
        selected, stats = allocate_candidates(pd.DataFrame([candidate()]), profile, costs, 100)
        curve, risk = mark_to_market(selected, {"ADA-USD": history([100, 100, 100])}, costs, 2, 100)
        self.assertAlmostEqual(stats["ending_equity"], 98)
        self.assertEqual(list(curve.close_mtm_equity), [98, 98, 98])
        self.assertAlmostEqual(risk["close_mtm_drawdown_pct"], 2)

    def test_exit_candle_low_is_included_before_close(self):
        selected, _ = allocate_candidates(pd.DataFrame([candidate()]), replace(THREE_COIN_V2, position_fraction=0.5), ZERO_COST, 100)
        curve, risk = mark_to_market(selected, {"ADA-USD": history([100, 100, 100], [100, 100, 50])}, ZERO_COST, 2, 100)
        self.assertEqual(curve.adverse_hour_equity.iloc[-1], 50)
        self.assertAlmostEqual(risk["adverse_hour_drawdown_pct"], 50)

    def test_intrabar_low_is_not_compared_to_a_later_new_close_peak(self):
        selected, _ = allocate_candidates(pd.DataFrame([candidate(exit_price=200)]), replace(THREE_COIN_V2, position_fraction=0.5), ZERO_COST, 100)
        _, risk = mark_to_market(selected, {"ADA-USD": history([100, 100, 200], [1, 100, 100])}, ZERO_COST, 2, 100)
        self.assertEqual(risk["adverse_hour_drawdown_pct"], 0)

    def test_missing_hour_is_disclosed(self):
        selected, _ = allocate_candidates(pd.DataFrame([candidate()]), THREE_COIN_V2, ZERO_COST, 100)
        _, risk = mark_to_market(selected, {"ADA-USD": history([100, 100, 100]).iloc[[0, 2]]}, ZERO_COST, 2, 100)
        self.assertEqual(risk["missing_open_position_mark_hours"], 1)

    def test_elapsed_rollover_cost_gap_is_reported(self):
        trade = candidate(end="2024-01-01 04:00")
        trade["bars_held"] = 3
        costs = HistoricalCosts("margin", 0, 0, 0, 0.01, 0)
        _, stats = allocate_candidates(pd.DataFrame([trade]), replace(THREE_COIN_V2, position_fraction=0.5), costs, 100)
        self.assertAlmostEqual(stats["ending_equity"], 100)
        self.assertAlmostEqual(stats["extra_elapsed_rollover_cost_cash"], 1)

    @patch("ml_strategy_backtest.backtest_symbol")
    def test_regeneration_uses_shared_profile_settings(self, backtest):
        backtest.return_value = {"trades": [candidate()]}
        profile = replace(THREE_COIN_V2, symbols={"ADA-USD": THREE_COIN_V2.symbols["ADA-USD"]})
        regenerate_candidates(profile, {"ADA-USD": history([100, 100, 100])}, 4000, 720)
        args = backtest.call_args.kwargs
        self.assertEqual(args["horizon"], 24)
        self.assertEqual(args["buy_thr"], 0.70)
        self.assertEqual(args["exit_thr"], 0.40)
        self.assertFalse(args["use_dynamic_threshold"])
        self.assertEqual(args["train_min"], 4000)


if __name__ == "__main__":
    unittest.main()
