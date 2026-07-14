"""
LONGER-HORIZON ML STRATEGY - HONEST WALK-FORWARD BACKTEST (Step 2 of "build a real edge")

What research_edge.py told us
-----------------------------
There is no edge at the 1h timeframe the current bot trades, but a small, real
edge appears when predicting direction ~48 hours ahead (especially on XRP/ADA/SOL).

What this script does
---------------------
Turns that finding into an actual tradeable strategy and tests it *properly*:

1. Walk-forward: the model is retrained only on PAST data, then makes decisions
   on strictly later data. It is retrained periodically (like you would in real
   life), never using the future.
2. Non-overlapping trades: only one position at a time. Enter when the model is
   confident, hold for a fixed number of bars (a few days), then exit and look
   for the next trade. This is realistic and avoids the overlapping-window
   inflation that can flatter research numbers.
3. Realistic costs: maker fee on both sides, scaled by leverage. Optional ATR
   stop-loss to cut losing trades early.

It does NOT touch the live bot. It tells us whether this strategy is worth
promoting to live trading.
"""

import argparse
import sys
import numpy as np
import pandas as pd

# Reuse the same data + feature code so research and backtest stay consistent.
from backtest import get_history, compute_atr
from research_edge import build_labels, make_model
from ml_strategy import (
    KrakenCostModel,
    btc_regime_state,
    build_cost_aware_labels,
    build_enhanced_features,
    compute_btc_regime_frame,
    compute_strict_btc_bear_frame,
    dynamic_probability_threshold,
    estimate_payoff_stats,
    expected_value,
    get_strategy_profile,
    passes_fng_filter,
    recent_volatility,
    relative_strength_7d,
    strict_btc_bear_state,
)


LIVE_ML_SYMBOLS = "XRP-USD,ADA-USD,SOL-USD,LINK-USD,DOGE-USD"


def supplied_cli_options(argv: list[str]) -> set[str]:
    return {
        token.split("=", 1)[0]
        for token in argv
        if token.startswith("--")
    }


def option_was_supplied(options: set[str], name: str) -> bool:
    return name in options


def apply_live_profile_defaults(args: argparse.Namespace, supplied_options: set[str]) -> None:
    profile = get_strategy_profile(args.live_profile)

    if not option_was_supplied(supplied_options, "--symbols"):
        args.symbols = LIVE_ML_SYMBOLS
    if not option_was_supplied(supplied_options, "--horizon"):
        args.horizon = profile.horizon
    if not option_was_supplied(supplied_options, "--buy-thr"):
        args.buy_thr = profile.buy_thr
    if not option_was_supplied(supplied_options, "--sell-thr"):
        args.sell_thr = 0.0 if profile.long_only else 0.35
    if not option_was_supplied(supplied_options, "--exit-thr"):
        args.exit_thr = profile.exit_thr
    if not option_was_supplied(supplied_options, "--no-fng-features"):
        args.no_fng_features = not profile.use_fng_features
    if not option_was_supplied(supplied_options, "--fng-filter"):
        args.fng_filter = profile.use_fng_filter
    if profile.long_only and not option_was_supplied(supplied_options, "--long-only"):
        args.long_only = True
    if profile.version == "v3" and not option_was_supplied(supplied_options, "--v3"):
        args.v3 = True

    if not option_was_supplied(supplied_options, "--margin-open-fee"):
        args.margin_open_fee = profile.margin_open_fee
    if not option_was_supplied(supplied_options, "--rollover-fee"):
        args.rollover_fee = profile.rollover_fee_4h
    if not option_was_supplied(supplied_options, "--minimum-edge"):
        args.minimum_edge = profile.minimum_edge
    if not option_was_supplied(supplied_options, "--ev-cost-multiplier"):
        args.ev_cost_multiplier = profile.ev_cost_multiplier
    if not option_was_supplied(supplied_options, "--cost-aware-labels"):
        args.cost_aware_labels = profile.use_cost_aware_labels
    if not option_was_supplied(supplied_options, "--btc-features"):
        args.btc_features = profile.use_btc_features
    if not option_was_supplied(supplied_options, "--btc-regime-filter"):
        args.btc_regime_filter = profile.use_btc_regime_filter
    if not option_was_supplied(supplied_options, "--relative-strength-filter"):
        args.relative_strength_filter = profile.use_relative_strength_filter
    if not option_was_supplied(supplied_options, "--expected-value-filter"):
        args.expected_value_filter = profile.use_expected_value_filter

    if not option_was_supplied(supplied_options, "--entry-fee"):
        args.entry_fee = args.maker_entry_fee
    if not option_was_supplied(supplied_options, "--exit-fee"):
        args.exit_fee = args.taker_exit_fee


def backtest_symbol(
    symbol: str,
    data: pd.DataFrame,
    horizon: int,
    buy_thr: float,
    sell_thr: float,
    fee_rate: float,
    leverage: float,
    model_name: str,
    train_min: int,
    retrain_every: int,
    atr_stop_mult: float,
    atr_period: int,
    exit_thr: float = 0.0,
    use_fng_features: bool = True,
    use_fng_filter: bool = False,
    starting_equity: float = 1000.0,
    margin_open_fee: float = 0.0002,
    rollover_fee: float = 0.0002,
    btc_data: pd.DataFrame | None = None,
    cost_model: KrakenCostModel | None = None,
    use_cost_aware_labels: bool = False,
    use_btc_features: bool = False,
    use_btc_regime_filter: bool = False,
    use_relative_strength_filter: bool = False,
    use_expected_value_filter: bool = False,
    ev_cost_multiplier: float = 1.5,
    entry_fee_rate: float | None = None,
    exit_fee_rate: float | None = None,
    bear_policy: str | None = None,
    regime_mode: str = 'default',
    use_dynamic_threshold: bool = True,
) -> dict:
    """Run the walk-forward backtest for one coin and return its stats.

    bear_policy (research):
      None / ignored  - normal long/short thresholds (plus optional regime filter)
      'cash'          - skip new entries while BTC is in bear/weak
      'short'         - long-only outside bear; short-only inside bear

    regime_mode (research):
      'default' - existing weak/strong/neutral BTC classifier
      'strict'  - sustained bear: below 200d EMA AND down >=10% over 8 weeks
    """
    cost_model = cost_model or KrakenCostModel(
        maker_entry_fee=fee_rate,
        taker_entry_fee=fee_rate,
        taker_exit_fee=fee_rate,
        margin_open_fee=margin_open_fee,
        margin_rollover_fee_4h=rollover_fee,
        spread_buffer=0.0,
        slippage_buffer=0.0,
        minimum_edge=0.0,
    )
    btc_for_features = btc_data if (use_btc_features and btc_data is not None) else None
    feats = build_enhanced_features(data, use_fng_features, btc_for_features)
    if use_cost_aware_labels:
        labels, _fwd = build_cost_aware_labels(data, horizon, cost_model)
    else:
        labels, _fwd = build_labels(data, horizon)

    feature_cols = list(feats.columns)
    X_all = feats.values
    y_all = labels.values
    close = data['Close'].values
    high = data['High'].values
    low = data['Low'].values
    atr_all = compute_atr(data, atr_period).values if atr_stop_mult > 0 else None
    if btc_data is None:
        btc_regimes = pd.DataFrame()
    elif regime_mode == 'strict':
        btc_regimes = compute_strict_btc_bear_frame(btc_data)
    else:
        btc_regimes = compute_btc_regime_frame(btc_data)
    rs_7d = relative_strength_7d(data, btc_data) if btc_data is not None else None

    n = len(data)
    # A row is usable as a feature only if none of its features are NaN.
    valid_feat = ~np.isnan(X_all).any(axis=1)

    entry_fee_rate = fee_rate if entry_fee_rate is None else entry_fee_rate
    exit_fee_rate = fee_rate if exit_fee_rate is None else exit_fee_rate
    fee_cost_pct = (entry_fee_rate + exit_fee_rate) * 100 * leverage

    model = None
    last_train_pos = -10**9
    last_train_mask = None

    trades = []
    equity = starting_equity
    equity_curve = [equity]

    # Start once we have enough history to train the first model.
    i = train_min
    oos_start_price = close[i] if i < n else close[-1]

    while i < n - 1:
        if not valid_feat[i]:
            i += 1
            continue

        # ---- Retrain periodically, using ONLY data whose label is known ----
        # A sample at position j only has a valid label once we are at least
        # `horizon` bars past it, so we train on rows with index <= i - horizon.
        if model is None or (i - last_train_pos) >= retrain_every:
            train_upto = i - horizon
            if train_upto > train_min // 2:
                mask = np.zeros(n, dtype=bool)
                mask[:train_upto + 1] = True
                mask &= valid_feat & ~np.isnan(y_all)
                if mask.sum() >= 300 and len(np.unique(y_all[mask])) > 1:
                    model = make_model(model_name)
                    model.fit(X_all[mask], y_all[mask])
                    last_train_pos = i
                    last_train_mask = mask

        if model is None:
            i += 1
            continue

        # ---- Decide whether to enter ----
        prob_up = float(model.predict_proba(X_all[i:i + 1])[:, 1][0])

        estimated_cost = cost_model.estimated_total_cost(horizon, 'maker')
        avg_win, avg_loss = estimate_payoff_stats(_fwd, last_train_mask) if last_train_mask is not None else (0.0, 0.0)
        if use_dynamic_threshold:
            dyn_thr = dynamic_probability_threshold(buy_thr, avg_win, avg_loss, estimated_cost)
        else:
            dyn_thr = float(buy_thr)
        ev = expected_value(prob_up, avg_win, avg_loss, estimated_cost)

        regime_name = 'unknown'
        in_bear = False
        if btc_data is not None and not btc_regimes.empty:
            if regime_mode == 'strict':
                regime = strict_btc_bear_state(btc_regimes, data.index[i])
            else:
                regime = btc_regime_state(btc_regimes, data.index[i])
            regime_name = regime.regime
            in_bear = regime.block_new_entries
            # cash-in-bear: skip entries while BTC is in bear/weak
            if (use_btc_regime_filter or bear_policy == 'cash') and in_bear:
                i += 1
                equity_curve.append(equity)
                continue

        rel_strength = 0.0
        if rs_7d is not None:
            rel_strength = float(rs_7d.iloc[i]) if not np.isnan(rs_7d.iloc[i]) else 0.0
            if use_relative_strength_filter and rel_strength <= 0:
                i += 1
                equity_curve.append(equity)
                continue

        if use_expected_value_filter and not (
            prob_up > dyn_thr and ev > 0 and ev > ev_cost_multiplier * estimated_cost
        ):
            # EV filter is long-edge oriented; for short-in-bear shorts, skip it.
            if not (bear_policy == 'short' and in_bear):
                i += 1
                equity_curve.append(equity)
                continue

        if bear_policy == 'short':
            # Non-bear: long only. Bear: short only.
            if in_bear:
                if sell_thr > 0 and prob_up < sell_thr:
                    direction = 'short'
                else:
                    i += 1
                    equity_curve.append(equity)
                    continue
            else:
                if prob_up > dyn_thr:
                    direction = 'long'
                else:
                    i += 1
                    equity_curve.append(equity)
                    continue
        elif prob_up > dyn_thr:
            direction = 'long'
        elif sell_thr > 0 and prob_up < sell_thr:
            direction = 'short'
        else:
            i += 1
            equity_curve.append(equity)
            continue

        if direction == 'long' and use_fng_filter:
            if not passes_fng_filter(data.index[i]):
                i += 1
                equity_curve.append(equity)
                continue

        entry_price = close[i]
        entry_atr = atr_all[i] if atr_all is not None else None

        # Precompute stop level (optional).
        if entry_atr is not None and not np.isnan(entry_atr) and entry_atr > 0:
            if direction == 'long':
                stop_price = entry_price - atr_stop_mult * entry_atr
            else:
                stop_price = entry_price + atr_stop_mult * entry_atr
        else:
            stop_price = None

        # ---- Hold for up to `horizon` bars; exit on stop, model bail, or time ----
        exit_idx = min(i + horizon, n - 1)
        exit_price = close[exit_idx]
        exit_reason = 'time'
        for j in range(i + 1, exit_idx + 1):
            if stop_price is not None:
                if direction == 'long' and low[j] <= stop_price:
                    exit_price, exit_idx, exit_reason = stop_price, j, 'stop'
                    break
                if direction == 'short' and high[j] >= stop_price:
                    exit_price, exit_idx, exit_reason = stop_price, j, 'stop'
                    break
            # Adaptive exit: model lost confidence during the hold.
            if exit_thr > 0 and direction == 'long' and valid_feat[j]:
                p = float(model.predict_proba(X_all[j:j + 1])[:, 1][0])
                if p < exit_thr:
                    exit_price, exit_idx, exit_reason = close[j], j, 'model_exit'
                    break
            if exit_thr > 0 and direction == 'short' and valid_feat[j]:
                # Bail on shorts if model flips back toward "up".
                p = float(model.predict_proba(X_all[j:j + 1])[:, 1][0])
                if p > (1.0 - exit_thr):
                    exit_price, exit_idx, exit_reason = close[j], j, 'model_exit'
                    break

        if direction == 'long':
            gross_pct = (exit_price - entry_price) / entry_price * 100 * leverage
        else:
            gross_pct = (entry_price - exit_price) / entry_price * 100 * leverage

        # Kraken margin costs: opening fee plus rollover every 4 hours.
        bars_held = exit_idx - i
        if use_cost_aware_labels or use_expected_value_filter:
            estimated_trade_cost = cost_model.estimated_total_cost(bars_held, 'maker')
            margin_cost_pct = estimated_trade_cost * 100 * leverage
            net_pct = gross_pct - margin_cost_pct
        else:
            margin_cost_pct = 0.0
            if leverage > 1:
                n_rollovers = bars_held // 4
                margin_cost_pct = (margin_open_fee + rollover_fee * n_rollovers) * 100 * leverage
            net_pct = gross_pct - fee_cost_pct - margin_cost_pct

        equity *= (1 + net_pct / 100)
        equity_curve.append(equity)

        trades.append({
            'symbol': symbol,
            'direction': direction,
            'btc_regime': regime_name,
            'prob_up': round(prob_up, 3),
            'dynamic_threshold': round(dyn_thr, 3),
            'expected_value': round(ev, 5),
            'estimated_cost': round(estimated_cost, 5),
            'relative_strength_7d': round(rel_strength, 5),
            'score': round(ev / recent_volatility(data.iloc[:i + 1]), 5),
            'entry_time': data.index[i],
            'exit_time': data.index[exit_idx],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'net_pnl_pct': net_pct,
            'margin_cost_pct': round(margin_cost_pct, 4),
            'exit_reason': exit_reason,
            'bars_held': bars_held,
        })

        i = exit_idx + 1  # non-overlapping: resume after the trade closes

    # Buy & hold benchmark over the same out-of-sample window.
    bh_return = (close[-1] / oos_start_price - 1) * 100

    return summarize(symbol, trades, equity_curve, starting_equity, bh_return)


def summarize(symbol, trades, equity_curve, starting_equity, bh_return) -> dict:
    """Compute the headline stats for one coin."""
    if not trades:
        return {'symbol': symbol, 'n_trades': 0, 'bh_return': bh_return}

    pnls = np.array([t['net_pnl_pct'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    peak = starting_equity
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak * 100 if peak > 0 else 0.0)

    gross_win = wins.sum()
    gross_loss = abs(losses.sum())

    return {
        'symbol': symbol,
        'n_trades': len(trades),
        'win_rate': (pnls > 0).mean() * 100,
        'avg_win': wins.mean() if len(wins) else 0.0,
        'avg_loss': losses.mean() if len(losses) else 0.0,
        'expectancy': pnls.mean(),
        'profit_factor': (gross_win / gross_loss) if gross_loss else float('inf'),
        'total_return': (equity_curve[-1] / equity_curve[0] - 1) * 100,
        'max_dd': max_dd,
        'bh_return': bh_return,
        'trades': trades,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the longer-horizon ML strategy.")
    parser.add_argument('--symbols', default='XRP-USD,ADA-USD,SOL-USD,BTC-USD,ETH-USD')
    parser.add_argument('--period', default='720d')
    parser.add_argument('--interval', default='1h')
    parser.add_argument('--live-profile', choices=['v2', 'v3'], default=None,
                        help="Apply saved live profile defaults and live symbol set unless explicitly overridden.")
    parser.add_argument('--horizon', type=int, default=72, help="Hold this many bars (~3 days at 1h).")
    parser.add_argument('--buy-thr', type=float, default=0.70)
    parser.add_argument('--sell-thr', type=float, default=0.45)
    parser.add_argument('--exit-thr', type=float, default=0.40,
                        help="Close long early if model probability drops below this (0=off).")
    parser.add_argument('--no-fng-features', action='store_true',
                        help="Disable Fear & Greed features.")
    parser.add_argument('--fng-filter', action='store_true',
                        help="Skip entries when F&G index is 25-40.")
    parser.add_argument('--fee', type=float, default=0.001, help="Maker fee per side (0.001 = 0.10%%).")
    parser.add_argument('--entry-fee', type=float, default=None,
                        help="Entry fee used for realized PnL. Defaults to --fee unless --live-profile is used.")
    parser.add_argument('--exit-fee', type=float, default=None,
                        help="Exit fee used for realized PnL. Defaults to --fee unless --live-profile is used.")
    parser.add_argument('--leverage', type=float, default=2.0)
    parser.add_argument('--model', choices=['logistic', 'gbm'], default='logistic')
    parser.add_argument('--train-min', type=int, default=4000, help="Bars before trading starts.")
    parser.add_argument('--retrain-every', type=int, default=720, help="Retrain cadence in bars.")
    parser.add_argument('--atr-stop-mult', type=float, default=0.0, help="ATR stop distance (0 = off).")
    parser.add_argument('--atr-period', type=int, default=14)
    parser.add_argument('--margin-open-fee', type=float, default=0.0002,
                        help="Kraken margin opening fee on position value (0.0002 = 0.02%%).")
    parser.add_argument('--rollover-fee', type=float, default=0.0002,
                        help="Kraken rollover fee per 4h held, on position value (0.0002 = 0.02%%).")
    parser.add_argument('--long-only', action='store_true',
                        help="Ignore short signals (sets sell threshold to 0).")
    parser.add_argument('--v3', action='store_true',
                        help="Use V3 profile (cost-aware labels). Default is V2-style labels.")
    parser.add_argument('--cost-aware-labels', action='store_true')
    parser.add_argument('--btc-features', action='store_true')
    parser.add_argument('--btc-regime-filter', action='store_true')
    parser.add_argument('--relative-strength-filter', action='store_true')
    parser.add_argument('--expected-value-filter', action='store_true')
    parser.add_argument('--no-dynamic-threshold', action='store_true',
                        help="Use --buy-thr exactly instead of raising it to the breakeven threshold.")
    parser.add_argument('--maker-entry-fee', type=float, default=0.0023)
    parser.add_argument('--taker-entry-fee', type=float, default=0.0040)
    parser.add_argument('--taker-exit-fee', type=float, default=0.0040)
    parser.add_argument('--spread-buffer', type=float, default=0.0005)
    parser.add_argument('--slippage-buffer', type=float, default=0.0010)
    parser.add_argument('--minimum-edge', type=float, default=0.0075)
    parser.add_argument('--ev-cost-multiplier', type=float, default=1.5)
    parser.add_argument('--out', default='ml_strategy_trades.csv')
    args = parser.parse_args()
    supplied_options = supplied_cli_options(sys.argv[1:])

    if args.live_profile and args.v3 and args.live_profile != 'v3':
        parser.error("--v3 conflicts with --live-profile v2")
    if args.live_profile:
        apply_live_profile_defaults(args, supplied_options)

    if args.long_only:
        args.sell_thr = 0.0
    if args.v3:
        args.cost_aware_labels = True
        if args.margin_open_fee == parser.get_default('margin_open_fee'):
            args.margin_open_fee = 0.0004
        if args.rollover_fee == parser.get_default('rollover_fee'):
            args.rollover_fee = 0.0004

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    cost_model = KrakenCostModel(
        maker_entry_fee=args.maker_entry_fee,
        taker_entry_fee=args.taker_entry_fee,
        taker_exit_fee=args.taker_exit_fee,
        margin_open_fee=args.margin_open_fee,
        margin_rollover_fee_4h=args.rollover_fee,
        spread_buffer=args.spread_buffer,
        slippage_buffer=args.slippage_buffer,
        minimum_edge=args.minimum_edge,
    )
    needs_btc = args.btc_features or args.btc_regime_filter or args.relative_strength_filter
    btc_data = get_history('BTC-USD', args.period, args.interval) if needs_btc else None

    entry_fee = args.fee if args.entry_fee is None else args.entry_fee
    exit_fee = args.fee if args.exit_fee is None else args.exit_fee
    if entry_fee == exit_fee:
        fee_label = f"fee={entry_fee * 100:.2f}%/side"
    else:
        fee_label = f"entry_fee={entry_fee * 100:.2f}% exit_fee={exit_fee * 100:.2f}%"

    profile_label = f" | live_profile={args.live_profile}" if args.live_profile else ""
    print(f"ML strategy backtest{profile_label} | model={args.model} | horizon={args.horizon}b | "
          f"lev={args.leverage}x | {fee_label} | "
          f"margin open={args.margin_open_fee * 100:.2f}% + rollover={args.rollover_fee * 100:.2f}%/4h | "
          f"buy>{args.buy_thr} sell<{args.sell_thr} | exit_thr={args.exit_thr or 'off'} | "
          f"fng={'filter' if args.fng_filter else 'features' if not args.no_fng_features else 'off'} | "
          f"atr_stop={args.atr_stop_mult or 'off'}")
    if args.v3 or args.cost_aware_labels or args.expected_value_filter:
        print(f"V3 gates | cost_labels={args.cost_aware_labels} btc_features={args.btc_features} "
              f"btc_regime={args.btc_regime_filter} rs_filter={args.relative_strength_filter} "
              f"ev_filter={args.expected_value_filter} min_edge={args.minimum_edge * 100:.2f}% "
              f"spread={args.spread_buffer * 100:.2f}% slippage={args.slippage_buffer * 100:.2f}%")
    print("Running walk-forward (retrain on past only)...\n")

    results = []
    all_trades = []
    for symbol in symbols:
        data = get_history(symbol, args.period, args.interval)
        if data.empty or len(data) <= args.train_min + args.horizon + 50:
            print(f"  {symbol}: not enough data, skipping")
            continue
        r = backtest_symbol(
            symbol, data, args.horizon, args.buy_thr, args.sell_thr, args.fee,
            args.leverage, args.model, args.train_min, args.retrain_every,
            args.atr_stop_mult, args.atr_period,
            exit_thr=args.exit_thr,
            use_fng_features=not args.no_fng_features,
            use_fng_filter=args.fng_filter,
            margin_open_fee=args.margin_open_fee, rollover_fee=args.rollover_fee,
            btc_data=btc_data,
            cost_model=cost_model,
            use_cost_aware_labels=args.cost_aware_labels,
            use_btc_features=args.btc_features,
            use_btc_regime_filter=args.btc_regime_filter,
            use_relative_strength_filter=args.relative_strength_filter,
            use_expected_value_filter=args.expected_value_filter,
            ev_cost_multiplier=args.ev_cost_multiplier,
            entry_fee_rate=entry_fee,
            exit_fee_rate=exit_fee,
            use_dynamic_threshold=not args.no_dynamic_threshold,
        )
        results.append(r)
        all_trades.extend(r.get('trades', []))
        if r['n_trades'] == 0:
            print(f"  {symbol:9s} no trades (model never confident enough)")
        else:
            print(f"  {symbol:9s} trades={r['n_trades']:4d}  win={r['win_rate']:4.1f}%  "
                  f"exp/trade={r['expectancy']:+.2f}%  PF={r['profit_factor']:.2f}  "
                  f"return={r['total_return']:+7.1f}%  (buy&hold {r['bh_return']:+.1f}%)  "
                  f"maxDD={r['max_dd']:.1f}%")

    pooled = [t['net_pnl_pct'] for t in all_trades]
    print("\n" + "=" * 70)
    print("OVERALL (all coins pooled)")
    print("=" * 70)
    if not pooled:
        print("  No trades. Try lowering the confidence thresholds.")
        return
    pooled = np.array(pooled)
    wins = pooled[pooled > 0]
    losses = pooled[pooled <= 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float('inf')
    print(f"  Trades:        {len(pooled)}")
    print(f"  Win rate:      {(pooled > 0).mean() * 100:.1f}%")
    print(f"  Expectancy:    {pooled.mean():+.2f}% per trade (after fees)")
    print(f"  Profit factor: {'inf' if pf == float('inf') else f'{pf:.2f}'}")
    reasons = {}
    for t in all_trades:
        reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1
    print("  Exit reasons:  " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    print("=" * 70)

    # ---- Temporal robustness: does the edge hold in EACH time window, or is it
    # one lucky period? We split all trades into 3 equal time windows by entry
    # date and report each. A real edge should be positive in most windows. ----
    sorted_trades = sorted(all_trades, key=lambda t: t['entry_time'])
    if len(sorted_trades) >= 30:
        print("\nTemporal robustness (trades split into 3 time windows):")
        third = len(sorted_trades) // 3
        windows = [sorted_trades[:third], sorted_trades[third:2 * third], sorted_trades[2 * third:]]
        for w_idx, w in enumerate(windows, 1):
            if not w:
                continue
            wp = np.array([t['net_pnl_pct'] for t in w])
            start = pd.Timestamp(w[0]['entry_time']).date()
            end = pd.Timestamp(w[-1]['entry_time']).date()
            wl = wp[wp <= 0]
            wpf = (wp[wp > 0].sum() / abs(wl.sum())) if len(wl) and wl.sum() != 0 else float('inf')
            print(f"  Window {w_idx} [{start}..{end}]: n={len(w):4d}  "
                  f"win={ (wp > 0).mean() * 100:4.1f}%  exp={wp.mean():+.2f}%  "
                  f"PF={'inf' if wpf == float('inf') else f'{wpf:.2f}'}")
        print("=" * 70)

    if all_trades:
        pd.DataFrame(all_trades).to_csv(args.out, index=False)
        print(f"\nSaved {len(all_trades)} trades to {args.out}")


if __name__ == "__main__":
    main()
