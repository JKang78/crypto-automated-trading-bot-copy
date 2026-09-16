"""Compare two virtual portfolios using public data; no order/account API exists here.

Signals use completed Yahoo hourly candles and the shared V2 model. Executions
are simulated at Kraken bid/ask plus slippage, with taker fees on both sides.
This deliberately does not assume that hypothetical maker orders would fill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path

import pandas as pd
import requests

from backtest import get_history
from ml_strategy import KrakenCostModel
from portfolio_profiles import build_strategies, get_portfolio_profile


PORTFOLIOS = ('three_coin_v2', 'six_coin_v2')
PAIRS = {'ADA-USD': 'ADAUSD', 'DOGE-USD': 'XDGUSD', 'LINK-USD': 'LINKUSD',
         'SOL-USD': 'SOLUSD', 'XLM-USD': 'XLMUSD', 'XRP-USD': 'XRPUSD'}
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExecutionCosts:
    # Tier-1 taker schedule checked 2026-09-16; no account-tier discounts assumed.
    # https://www.kraken.com/features/fee-schedule
    taker_fee: float = 0.008
    margin_open_fee: float = 0.0004
    rollover_fee_4h: float = 0.0004
    slippage_per_side: float = 0.0005


def utc_naive(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert('UTC').tz_localize(None) if ts.tzinfo else ts


def completed_history(data: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Exclude the still-forming hourly candle; fail visibly on stale inputs."""
    data = data.copy()
    data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
    data = data.sort_index()
    data = data[~data.index.duplicated(keep='last')]
    data = data[data.index + pd.Timedelta(hours=1) <= now]
    if data.empty or now - (data.index[-1] + pd.Timedelta(hours=1)) > pd.Timedelta(hours=3):
        raise ValueError('Missing or stale completed hourly market data')
    values = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    if not all(math.isfinite(float(x)) for x in values.iloc[-1]):
        raise ValueError('Non-finite market data')
    if (values[['Open', 'High', 'Low', 'Close']].iloc[-1] <= 0).any():
        raise ValueError('Non-positive market price')
    return data


def public_get(endpoint: str, pairs: str) -> dict:
    if endpoint not in ('AssetPairs', 'Ticker'):
        raise ValueError('Only public market endpoints are permitted')
    response = requests.get('https://api.kraken.com/0/public/' + endpoint,
                            params={'pair': pairs}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get('error'):
        raise RuntimeError(f"Kraken public data error: {payload['error']}")
    return payload['result']


def fetch_market() -> tuple[dict, dict]:
    names = ','.join(PAIRS.values())
    metadata = public_get('AssetPairs', names)
    tickers = public_get('Ticker', names)
    rules, quotes = {}, {}
    for symbol, altname in PAIRS.items():
        key, rule = next(((k, v) for k, v in metadata.items()
                          if v.get('altname') == altname), (None, None))
        if key is None:
            raise ValueError(f'Missing pair metadata: {symbol}')
        ticker = tickers.get(key, tickers.get(altname))
        if not ticker:
            raise ValueError(f'Missing ticker: {symbol}')
        bid, ask = float(ticker['b'][0]), float(ticker['a'][0])
        if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
            raise ValueError(f'Invalid bid/ask: {symbol}')
        rules[symbol] = rule
        quotes[symbol] = {'bid': bid, 'ask': ask}
    return rules, quotes


def configuration(start_equity: float, costs: ExecutionCosts) -> dict:
    return {'start_equity_usd': start_equity, 'execution_costs': asdict(costs),
            'signal_costs': asdict(signal_cost_model()),
            'profiles': {name: {
                'symbols': {symbol: asdict(spec) for symbol, spec in
                            get_portfolio_profile(name).symbols.items()},
                'leverage': get_portfolio_profile(name).leverage,
                'max_open': get_portfolio_profile(name).max_open,
                'position_fraction': get_portfolio_profile(name).position_fraction,
                'use_dynamic_threshold': get_portfolio_profile(name).use_dynamic_threshold,
            } for name in PORTFOLIOS}}


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def signal_cost_model() -> KrakenCostModel:
    # Retain the existing model's inputs for comparison; execution P&L below
    # charges the higher current entry-level taker rate. These are separate.
    return KrakenCostModel(margin_open_fee=0.0002, margin_rollover_fee_4h=0.0002,
                           minimum_edge=0.0)


def new_state(start_equity: float, costs: ExecutionCosts) -> dict:
    if not math.isfinite(start_equity) or start_equity <= 0:
        raise ValueError('Starting paper equity must be finite and positive')
    config = configuration(start_equity, costs)
    return {'schema_version': SCHEMA_VERSION, 'mode': 'paper_comparison',
            'currency': 'USD', 'config': config, 'config_hash': config_hash(config),
            'accounts': {name: {
                'cash': start_equity, 'start_equity': start_equity,
                'open': {}, 'closed': [], 'last_signal_bar': {},
                'peak_equity': start_equity, 'max_drawdown_pct': 0.0,
                'observations': 0, 'utilization_sum': 0.0,
            } for name in PORTFOLIOS}}


def validate_output_path(path: Path, suffix: str) -> None:
    if path.is_symlink() or not path.name.startswith('paper_comparison') or path.suffix != suffix:
        raise ValueError(f'Use a separate paper_comparison*{suffix} output file: {path}')


def load_state(path: Path, start_equity: float, costs: ExecutionCosts) -> dict:
    validate_output_path(path, '.json')
    expected = new_state(start_equity, costs)
    if not path.exists():
        return expected
    state = json.loads(path.read_text())  # Corruption must never silently reset an account.
    for key in ('schema_version', 'mode', 'currency', 'config_hash', 'config'):
        if state.get(key) != expected[key]:
            raise ValueError(f'Paper state {key} mismatch; use a new comparison file')
    if set(state['accounts']) != set(PORTFOLIOS):
        raise ValueError('Paper account set mismatch')
    for account in state['accounts'].values():
        if not math.isfinite(account['cash']):
            raise ValueError('Invalid paper cash balance')
    return state


def save_state(path: Path, state: dict) -> None:
    validate_output_path(path, '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def exit_value(position: dict, bid: float, now: pd.Timestamp,
               costs: ExecutionCosts) -> tuple[float, float, float]:
    price = bid * (1 - costs.slippage_per_side)
    hours = max(0.0, (now - utc_naive(position['entry_time'])).total_seconds() / 3600)
    rollover = position['notional_usd'] * costs.rollover_fee_4h * int(hours // 4)
    exit_fee = position['volume'] * price * costs.taker_fee
    change = position['volume'] * (price - position['entry_price']) - rollover - exit_fee
    return change, price, rollover + exit_fee


def mark_equity(account: dict, quotes: dict, now: pd.Timestamp, costs: ExecutionCosts) -> float:
    return account['cash'] + sum(exit_value(pos, quotes[symbol]['bid'], now, costs)[0]
                                 for symbol, pos in account['open'].items())


def observe(account: dict, quotes: dict, now: pd.Timestamp, costs: ExecutionCosts) -> None:
    equity = mark_equity(account, quotes, now, costs)
    reserved = sum(p['margin_usd'] for p in account['open'].values())
    account['marked_equity'] = equity
    account['reserved_margin'] = reserved
    account['peak_equity'] = max(account['peak_equity'], equity)
    dd = 100 * (1 - equity / account['peak_equity'])
    account['max_drawdown_pct'] = max(account['max_drawdown_pct'], dd)


def volume_for_trade(equity: float, free: float, price: float, exit_price: float, profile,
                     rule: dict, costs: ExecutionCosts) -> float:
    """Enforce exchange minimums, lot precision and margin plus opening fees."""
    step = Decimal(1).scaleb(-int(rule['lot_decimals']))
    minimum = max(float(rule['ordermin']), float(rule.get('costmin', 0)) / price)
    min_volume = Decimal(str(minimum)).quantize(step, rounding=ROUND_UP)
    requested = max(equity * profile.position_fraction * profile.leverage / price,
                    float(min_volume))
    fee = costs.taker_fee + costs.margin_open_fee
    # Reserve the immediate spread/slippage loss and exit fee as well; otherwise
    # a minimum-size bump can consume more margin than the marked account has.
    cost_per_unit = price * fee + price - exit_price + exit_price * costs.taker_fee
    affordable = max(0.0, free) / (price / profile.leverage + cost_per_unit)
    volume = Decimal(str(min(requested, affordable))).quantize(step, rounding=ROUND_DOWN)
    return float(volume) if volume >= min_volume else 0.0


def run_cycle(state: dict, histories: dict, quotes: dict, rules: dict,
              now: pd.Timestamp, costs: ExecutionCosts, strategies: dict | None = None) -> list[str]:
    """Advance both portfolios on the same snapshot; injected strategies aid tests."""
    now = utc_naive(now)
    if state.get('updated_at') and now <= utc_naive(state['updated_at']):
        return ['Snapshot already processed; no changes.']
    histories = {s: completed_history(histories[s], now) for s in PAIRS}
    model_costs = signal_cost_model()
    if strategies is None:
        strategies = {name: build_strategies(get_portfolio_profile(name), model_costs)
                      for name in PORTFOLIOS}
    actions = []
    if state.get('updated_at'):
        gap = (now - utc_naive(state['updated_at'])).total_seconds() / 3600
        if gap > 1:
            actions.append(f'Scan gap {gap:.1f}h; missed entries/exits are not backfilled.')
    for name in PORTFOLIOS:
        account, profile = state['accounts'][name], get_portfolio_profile(name)
        observe(account, quotes, now, costs)  # Capture losses while positions are still open.
        for symbol, pos in list(account['open'].items()):
            if rules[symbol].get('status') != 'online':
                actions.append(f'{name}: cannot simulate close on offline market {symbol}')
                continue
            strategy = strategies[name][symbol]
            due = now >= utc_naive(pos['exit_due'])
            early, prob = (False, None) if due else strategy.should_exit_early(histories[symbol])
            if not (due or early):
                continue
            change, price, closing_cost = exit_value(pos, quotes[symbol]['bid'], now, costs)
            account['cash'] += change
            pnl = change - pos['opening_cost_usd']
            account['closed'].append({**pos, 'symbol': symbol, 'exit_time': now.isoformat(),
                                      'exit_price': price, 'exit_reason': 'time' if due else 'model',
                                      'exit_probability': prob, 'pnl_usd': pnl,
                                      'total_cost_usd': closing_cost + pos['opening_cost_usd']})
            del account['open'][symbol]
            actions.append(f'{name}: PAPER CLOSE {symbol}, net ${pnl:+.2f}')

        candidates = []
        for symbol, spec in profile.symbols.items():
            bar = histories[symbol].index[-1].isoformat()
            if symbol in account['open'] or account['last_signal_bar'].get(symbol) == bar:
                continue
            account['last_signal_bar'][symbol] = bar
            sig = strategies[name][symbol].get_signal(histories[symbol])
            threshold = sig.dynamic_threshold or spec.buy_thr
            if sig.signal != 'BUY':
                reason = sig.blocked_reason or ('model unavailable' if not sig.dynamic_threshold
                                                else 'below entry threshold')
                actions.append(f'{name}: {symbol} {reason}, p={sig.prob_up:.3f}, threshold={threshold:.2f}')
                continue
            if not all(math.isfinite(float(v)) for v in (sig.prob_up, sig.score)):
                raise ValueError(f'Invalid model signal for {symbol}')
            candidates.append((symbol, sig))
        candidates.sort(key=lambda item: (-item[1].score, item[0]))
        for symbol, sig in candidates:
            if len(account['open']) >= profile.max_open:
                actions.append(f'{name}: skip {symbol}, all {profile.max_open} slots occupied')
                continue
            rule = rules[symbol]
            if rule.get('status') != 'online' or profile.leverage not in rule.get('leverage_buy', []):
                actions.append(f'{name}: skip {symbol}, requested margin market unavailable')
                continue
            equity = mark_equity(account, quotes, now, costs)
            free = equity - sum(p['margin_usd'] for p in account['open'].values())
            if equity <= 0 or free <= 0:
                actions.append(f'{name}: skip {symbol}, no free paper margin')
                continue
            price = quotes[symbol]['ask'] * (1 + costs.slippage_per_side)
            exit_price = quotes[symbol]['bid'] * (1 - costs.slippage_per_side)
            volume = volume_for_trade(equity, free, price, exit_price, profile, rule, costs)
            if volume <= 0:
                actions.append(f'{name}: skip {symbol}, exchange minimum exceeds free margin ${free:.2f}')
                continue
            notional = volume * price
            opening_cost = notional * (costs.taker_fee + costs.margin_open_fee)
            account['cash'] -= opening_cost
            spec = profile.symbols[symbol]
            account['open'][symbol] = {
                'entry_time': now.isoformat(), 'entry_price': price, 'volume': volume,
                'notional_usd': notional, 'margin_usd': notional / profile.leverage,
                'opening_cost_usd': opening_cost, 'horizon_hours': spec.horizon,
                'buy_threshold': spec.buy_thr, 'exit_threshold': spec.exit_thr,
                'exit_due': (now + pd.Timedelta(hours=spec.horizon)).isoformat(),
                'signal_bar': histories[symbol].index[-1].isoformat(),
                'prob_up': sig.prob_up, 'score': sig.score, 'execution': 'simulated_taker',
            }
            actions.append(f'{name}: PAPER OPEN {symbol}, {spec.horizon}h, margin ${notional/profile.leverage:.2f}')
        observe(account, quotes, now, costs)
        account['observations'] += 1
        equity = account['marked_equity']
        account['utilization_sum'] += account['reserved_margin'] / equity if equity > 0 else 0.0
    state.setdefault('started_at', now.isoformat())
    state['updated_at'] = now.isoformat()
    state['last_actions'] = actions
    return actions


def render_report(state: dict) -> str:
    lines = ['# Six-coin paper comparison', '',
             'Virtual USD accounts only. No exchange orders or real-money performance.', '',
             f"Observed at: {state['updated_at']} UTC; started: {state['started_at']} UTC.", '',
             '| Portfolio | Equity incl. open P&L | Return | Closed trades | Open | Observed drawdown |',
             '|---|---:|---:|---:|---:|---:|']
    for name, account in state['accounts'].items():
        ret = (account['marked_equity'] / account['start_equity'] - 1) * 100
        lines.append(f"| {name} | ${account['marked_equity']:.2f} | {ret:+.2f}% | "
                     f"{len(account['closed'])} | {len(account['open'])} | {account['max_drawdown_pct']:.2f}% |")
    lines += ['', 'Both portfolios use the same starting equity, data and execution costs.',
              'Simulated buys use current ask + slippage; sells use bid - slippage. '
              'Taker fees, opening margin fees and elapsed 4-hour rollover estimates are included.',
              'Paper fills charge 0.80% per side (Tier 1, checked 2026-09-16). '
              'Signal estimates retain the existing model cost settings; they are not realized returns.',
              'Drawdown includes open P&L at observed scans, but can miss losses between scans. '
              'No liquidation engine, account eligibility checks, or guaranteed fills are modeled.',
              'Signals use completed hourly bars; gaps are not retroactively traded. '
              'Each symbol is considered once per completed bar. There is no automatic promotion to live.', '',
              '## Latest decisions', '']
    lines += ['- ' + a for a in state.get('last_actions', [])] or ['- No new hourly signals.']
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-file', type=Path, default=Path('paper_comparison_state.json'))
    parser.add_argument('--report-file', type=Path, default=Path('paper_comparison_report.md'))
    parser.add_argument('--start-equity', type=float, default=170.0, help='Virtual USD equity per portfolio')
    args = parser.parse_args()
    validate_output_path(args.report_file, '.md')
    costs = ExecutionCosts()
    state = load_state(args.state_file, args.start_equity, costs)
    histories = {symbol: get_history(symbol, '720d', '1h') for symbol in PAIRS}
    rules, quotes = fetch_market()
    now = utc_naive(pd.Timestamp.now(tz='UTC'))
    run_cycle(state, histories, quotes, rules, now, costs)
    save_state(args.state_file, state)
    report = render_report(state)
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(report)
    print(report)


if __name__ == '__main__':
    main()
