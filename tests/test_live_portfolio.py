import unittest
from unittest.mock import patch

import ml_live_trade
from portfolio_profiles import SIX_COIN_V2


class LivePortfolioTests(unittest.TestCase):
    def test_live_factory_uses_every_six_coin_rule(self):
        with patch.object(ml_live_trade, 'PORTFOLIO', SIX_COIN_V2), \
             patch.object(ml_live_trade, 'PORTFOLIO_NAME', SIX_COIN_V2.name), \
             patch.object(ml_live_trade, 'SYMBOLS', list(SIX_COIN_V2.symbols)):
            strategies = ml_live_trade.build_live_strategies()

        self.assertEqual(set(strategies), set(SIX_COIN_V2.symbols))
        for symbol, definition in SIX_COIN_V2.symbols.items():
            strategy = strategies[symbol]
            self.assertEqual(strategy.horizon, definition.horizon)
            self.assertEqual(strategy.buy_thr, definition.buy_thr)
            self.assertEqual(strategy.exit_thr, definition.exit_thr)
            self.assertTrue(strategy.long_only)
            self.assertTrue(strategy.use_dynamic_threshold)

    def test_expected_live_allocation_and_position_cap(self):
        self.assertEqual(SIX_COIN_V2.position_fraction, 0.33)
        self.assertEqual(SIX_COIN_V2.max_open, 3)
        self.assertEqual(SIX_COIN_V2.leverage, 2)


if __name__ == '__main__':
    unittest.main()
