import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ml_strategy import MLSignal
from ml_portfolio_paper import (
    PAIRS, PORTFOLIOS, ExecutionCosts, completed_history, exit_value,
    load_state, mark_equity, new_state, public_get, run_cycle, save_state,
)
from portfolio_profiles import get_portfolio_profile


class FakeStrategy:
    def __init__(self, spec, buy=True):
        self.spec, self.buy = spec, buy

    def get_signal(self, data):
        return MLSignal('BUY' if self.buy else None, .6, .8, self.spec.horizon,
                        dynamic_threshold=self.spec.buy_thr, score=1.0)

    def should_exit_early(self, data):
        return False, .8


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp('2026-09-16T12:15:00')
        self.costs = ExecutionCosts()
        self.state = new_state(170, self.costs)
        frame = pd.DataFrame({'Open': [1., 1.], 'High': [1., 1.], 'Low': [1., 1.],
                              'Close': [1., 1.], 'Volume': [100., 100.]},
                             index=pd.to_datetime(['2026-09-16T11:00:00', '2026-09-16T12:00:00']))
        self.histories = {s: frame.copy() for s in PAIRS}
        self.quotes = {s: {'bid': .999, 'ask': 1.001} for s in PAIRS}
        self.rules = {s: {'ordermin': '1', 'costmin': '.5', 'lot_decimals': 8,
                          'status': 'online', 'leverage_buy': [2]} for s in PAIRS}
        self.strategies = {name: {s: FakeStrategy(spec) for s, spec in
                                 get_portfolio_profile(name).symbols.items()} for name in PORTFOLIOS}

    def cycle(self, now=None):
        return run_cycle(self.state, self.histories, self.quotes, self.rules,
                         now or self.now, self.costs, self.strategies)

    def test_public_transport_rejects_trading_endpoint_without_network(self):
        with patch('ml_portfolio_paper.requests.get') as request:
            with self.assertRaises(ValueError):
                public_get('AddOrder', 'ADAUSD')
            request.assert_not_called()

    def test_forming_candle_removed_and_stale_data_fails(self):
        result = completed_history(self.histories['ADA-USD'], self.now)
        self.assertEqual(result.index[-1], pd.Timestamp('2026-09-16T11:00:00'))
        with self.assertRaises(ValueError):
            completed_history(result, self.now + pd.Timedelta(hours=5))

    def test_per_symbol_horizons_account_isolation_and_slot_cap(self):
        self.cycle()
        for name, account in self.state['accounts'].items():
            self.assertEqual(len(account['open']), 3)
            self.assertLess(account['marked_equity'], 170)  # Fees/spread already recognized.
            self.assertGreaterEqual(account['marked_equity'], account['reserved_margin'])
            for symbol, position in account['open'].items():
                spec = get_portfolio_profile(name).symbols[symbol]
                self.assertEqual(pd.Timestamp(position['exit_due']) - self.now,
                                 pd.Timedelta(hours=spec.horizon))
        self.assertEqual(self.state['accounts']['six_coin_v2']['open']['DOGE-USD']['horizon_hours'], 48)
        self.assertEqual(self.state['accounts']['three_coin_v2']['open']['DOGE-USD']['horizon_hours'], 24)

    def test_same_snapshot_and_same_hour_do_not_duplicate_positions(self):
        self.cycle()
        initial = copy.deepcopy(self.state)
        self.cycle()
        self.assertEqual(initial, self.state)
        self.cycle(self.now + pd.Timedelta(minutes=15))
        for name in PORTFOLIOS:
            self.assertEqual(self.state['accounts'][name]['open'], initial['accounts'][name]['open'])
            self.assertEqual(self.state['accounts'][name]['cash'], initial['accounts'][name]['cash'])

    def test_unaffordable_top_candidate_does_not_suppress_other_coins(self):
        self.rules['ADA-USD']['ordermin'] = '1000000'
        self.cycle()
        six = self.state['accounts']['six_coin_v2']['open']
        self.assertNotIn('ADA-USD', six)
        self.assertEqual(len(six), 3)

    def test_minimum_bump_reserves_all_entry_and_exit_costs(self):
        self.rules['ADA-USD']['ordermin'] = '200'
        self.cycle()
        for account in self.state['accounts'].values():
            self.assertGreaterEqual(account['marked_equity'] + 1e-8, account['reserved_margin'])
            self.assertGreaterEqual(account['open']['ADA-USD']['volume'], 200)

    def test_holding_losses_update_drawdown_before_close(self):
        self.cycle()
        self.quotes = {s: {'bid': .5, 'ask': .501} for s in PAIRS}
        self.cycle(self.now + pd.Timedelta(minutes=15))
        for account in self.state['accounts'].values():
            self.assertGreater(account['max_drawdown_pct'], 50)
            self.assertEqual(len(account['closed']), 0)

    def test_time_exit_charges_costs_once_and_preserves_longer_horizon(self):
        self.cycle()
        baseline = self.state['accounts']['three_coin_v2']
        initial_cash = baseline['cash']
        later = self.now + pd.Timedelta(hours=24)
        expected = initial_cash + sum(exit_value(p, self.quotes[s]['bid'], later, self.costs)[0]
                                      for s, p in baseline['open'].items())
        for name in self.strategies:
            for strategy in self.strategies[name].values():
                strategy.buy = False
        for frame in self.histories.values():
            frame.index = frame.index + pd.Timedelta(hours=24)
        self.cycle(later)
        self.assertAlmostEqual(baseline['cash'], expected)
        self.assertEqual(len(baseline['closed']), 3)
        self.assertAlmostEqual(sum(p['pnl_usd'] for p in baseline['closed']), expected - 170)
        self.assertIn('DOGE-USD', self.state['accounts']['six_coin_v2']['open'])
        self.assertEqual(len(self.state['accounts']['six_coin_v2']['closed']), 1)

    def test_state_rejects_live_path_corruption_and_config_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paper_comparison_state.json'
            with self.assertRaises(ValueError):
                load_state(Path(directory) / 'ml_live_state.json', 170, self.costs)
            save_state(path, self.state)
            self.assertEqual(load_state(path, 170, self.costs), self.state)
            with self.assertRaises(ValueError):
                load_state(path, 171, self.costs)
            path.write_text('{broken')
            with self.assertRaises(json.JSONDecodeError):
                load_state(path, 170, self.costs)


if __name__ == '__main__':
    unittest.main()
