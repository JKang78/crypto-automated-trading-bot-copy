import unittest

import pandas as pd

from portfolio_profiles import get_portfolio_profile
from research_six_coin_validation import HistoricalCosts, allocate_candidates, mark_to_market


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.profile = get_portfolio_profile('six_coin_v2')
        self.costs = HistoricalCosts('zero', 0, 0, 0, 0, 0)
        self.times = pd.date_range('2024-01-01', periods=3, freq='h')

    def trade(self, symbol='ADA-USD', **changes):
        row = {'symbol': symbol, 'entry_time': self.times[0], 'exit_time': self.times[-1],
               'entry_price': 100., 'exit_price': 110., 'bars_held': 2,
               'score': 1., 'prob_up': .8, 'exit_reason': 'time'}
        row.update(changes)
        return row

    def history(self, closes, lows=None):
        return pd.DataFrame({'Close': closes, 'Low': lows or closes}, index=self.times)

    def simulate(self, trades, costs=None):
        return allocate_candidates(pd.DataFrame(trades), self.profile, costs or self.costs, 100.)

    def test_open_loss_is_visible_even_when_closed_trade_wins(self):
        selected, realized = self.simulate([self.trade()])
        curve, risk = mark_to_market(selected, {'ADA-USD': self.history([100, 60, 110])},
                                    self.costs, 2, 100)
        self.assertGreater(realized['return_pct'], 0)
        self.assertEqual(realized['realized_drawdown_pct'], 0)
        self.assertAlmostEqual(risk['close_mtm_drawdown_pct'], 26.4)
        self.assertAlmostEqual(curve.close_mtm_equity.iloc[-1], realized['ending_equity'])

    def test_position_cap_and_duplicate_symbols_reserve_margin(self):
        rows = [self.trade(s) for s in ('ADA-USD', 'ADA-USD', 'DOGE-USD', 'LINK-USD', 'SOL-USD')]
        selected, stats = self.simulate(rows)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected.symbol.nunique(), 3)
        self.assertLessEqual(stats['max_margin_fraction'], 1)
        self.assertEqual(stats['rejected_occupied_symbol'], 1)
        self.assertEqual(stats['rejected_position_cap'], 1)

    def test_entry_low_is_excluded_and_future_close_peak_is_not_used(self):
        selected, _ = self.simulate([self.trade()])
        _, risk = mark_to_market(selected, {'ADA-USD': self.history([100, 100, 110], [1, 100, 100])},
                                 self.costs, 2, 100)
        self.assertEqual(risk['adverse_hour_drawdown_pct'], 0)

    def test_missing_hour_is_counted_not_disguised_as_observed(self):
        selected, _ = self.simulate([self.trade()])
        data = self.history([100, 100, 110]).drop(self.times[1])
        _, risk = mark_to_market(selected, {'ADA-USD': data}, self.costs, 2, 100)
        self.assertEqual(risk['missing_open_position_mark_hours'], 1)

    def test_fee_recognition_does_not_double_charge_at_exit(self):
        costs = HistoricalCosts('fees', .01, .02, .001, .001, .001)
        selected, realized = self.simulate([self.trade()], costs)
        curve, _ = mark_to_market(selected, {'ADA-USD': self.history([100, 100, 110])},
                                  costs, 2, 100)
        expected = 100 + 33 * 2 * (.1 - .032)
        self.assertAlmostEqual(realized['ending_equity'], expected)
        self.assertAlmostEqual(curve.close_mtm_equity.iloc[-1], expected)

    def test_changed_cache_price_fails_validation(self):
        selected, _ = self.simulate([self.trade()])
        with self.assertRaises(ValueError):
            mark_to_market(selected, {'ADA-USD': self.history([99, 100, 110])}, self.costs, 2, 100)


if __name__ == '__main__':
    unittest.main()
