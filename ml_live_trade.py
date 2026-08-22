"""
REAL-MONEY LIVE TRADER for the longer-horizon ML strategy.

⚠️ THIS PLACES REAL ORDERS ON KRAKEN. ⚠️

The script is environment-driven. By default it runs one decision cycle, which
fits cron/GitHub Actions. Set ML_LIVE_RUN_FOREVER=true to keep it alive as a
24/7 process that wakes up every ML_LIVE_LOOP_INTERVAL_SEC seconds.

Safety design
-------------
- State (which positions we opened and when to close them) is saved to
  ml_live_state.json so it survives between independent cron runs.
- Before opening new live positions, the bot reads Kraken open positions so a
  missing state file does not cause duplicate exposure in the same market.
- Entries can use maker LIMIT orders or immediate MARKET orders depending on
  ML_LIVE_MAKER_ENTRY. Exits are always MARKET reduce-only orders.
- Long-only by default: the researched short side was not reliable enough.
- Set ML_LIVE_DRY_RUN=true to run the full logic WITHOUT placing real orders
  (read-only account calls only) - useful for testing.

This is separate from the old swing bot, which should be disabled so the two do
not fight over the same Kraken account.
"""

import os
import json
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

from kraken_bot_v4_advanced import Config, KrakenClient, Telegram
from backtest import get_history
from ml_strategy import (
    KrakenCostModel,
    MLSwingStrategy,
    btc_regime_state,
    build_cost_model,
    compute_btc_regime_frame,
    create_ml_strategy,
    expected_value,
    get_strategy_profile,
)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() == 'true'


# ─────────────────────────── Settings (env-overridable) ───────────────────────────
STRATEGY_VERSION = _env_str('ML_LIVE_STRATEGY', 'v2').lower()
PROFILE = get_strategy_profile(STRATEGY_VERSION)
STATE_FILE = _env_str('ML_LIVE_STATE_FILE', 'ml_live_state.json')
SYMBOLS = [s.strip() for s in _env_str(
    'ML_LIVE_SYMBOLS', 'SOL-USD,LINK-USD,DOGE-USD').split(',') if s.strip()]
PERIOD = _env_str('ML_LIVE_PERIOD', '720d')
INTERVAL = _env_str('ML_LIVE_INTERVAL', '1h')
HORIZON = _env_int('ML_LIVE_HORIZON', PROFILE.horizon)
BUY_THR = _env_float('ML_LIVE_BUY_THR', PROFILE.buy_thr)
SELL_THR = _env_float('ML_LIVE_SELL_THR', 0.0 if PROFILE.long_only else 0.35)
EXIT_THR = _env_float('ML_LIVE_EXIT_THR', PROFILE.exit_thr)
USE_FNG_FEATURES = _env_bool('ML_LIVE_FNG_FEATURES', PROFILE.use_fng_features)
USE_FNG_FILTER = _env_bool('ML_LIVE_FNG_FILTER', PROFILE.use_fng_filter)
LEVERAGE = _env_int('ML_LIVE_LEVERAGE', 2)
POSITION_FRACTION = _env_float('ML_LIVE_POSITION_FRACTION', 0.25)
MAX_OPEN = _env_int('ML_LIVE_MAX_OPEN', 3)
MARGIN_SAFETY_FACTOR = _env_float('ML_LIVE_MARGIN_SAFETY', 1.5)
# When True, bump a too-small 25% size up to Kraken's min order if affordable.
ALLOW_MIN_SIZE_BUMP = _env_bool('ML_LIVE_ALLOW_MIN_SIZE_BUMP', True)
DRY_RUN = _env_bool('ML_LIVE_DRY_RUN', False)
LONG_ONLY = _env_bool('ML_LIVE_LONG_ONLY', PROFILE.long_only)
USE_MAKER_ENTRY = _env_bool('ML_LIVE_MAKER_ENTRY', True)
MAKER_WAIT_SEC = _env_int('ML_LIVE_MAKER_WAIT_SEC', 90)
MAKER_ENTRY_FEE = _env_float('ML_LIVE_MAKER_ENTRY_FEE', 0.0023)
TAKER_ENTRY_FEE = _env_float('ML_LIVE_TAKER_ENTRY_FEE', 0.0040)
TAKER_EXIT_FEE = _env_float('ML_LIVE_TAKER_EXIT_FEE', 0.0040)
MARGIN_OPEN_FEE = _env_float('ML_LIVE_MARGIN_OPEN_FEE', PROFILE.margin_open_fee)
ROLLOVER_FEE_4H = _env_float('ML_LIVE_ROLLOVER_FEE_4H', PROFILE.rollover_fee_4h)
SPREAD_BUFFER = _env_float('ML_LIVE_SPREAD_BUFFER', 0.0005)
SLIPPAGE_BUFFER = _env_float('ML_LIVE_SLIPPAGE_BUFFER', 0.0010)
MINIMUM_EDGE = _env_float('ML_LIVE_MINIMUM_EDGE', PROFILE.minimum_edge)
EV_COST_MULTIPLIER = _env_float('ML_LIVE_EV_COST_MULTIPLIER', PROFILE.ev_cost_multiplier)
USE_COST_AWARE_LABELS = _env_bool('ML_LIVE_COST_AWARE_LABELS', PROFILE.use_cost_aware_labels)
USE_BTC_FEATURES = _env_bool('ML_LIVE_BTC_FEATURES', PROFILE.use_btc_features)
USE_BTC_REGIME_FILTER = _env_bool('ML_LIVE_BTC_REGIME_FILTER', PROFILE.use_btc_regime_filter)
USE_RELATIVE_STRENGTH_FILTER = _env_bool('ML_LIVE_RELATIVE_STRENGTH_FILTER', PROFILE.use_relative_strength_filter)
USE_EXPECTED_VALUE_FILTER = _env_bool('ML_LIVE_EXPECTED_VALUE_FILTER', PROFILE.use_expected_value_filter)
USE_EV_EXIT = _env_bool('ML_LIVE_EV_EXIT', PROFILE.use_ev_exit)
EV_GATED_MARKET_FALLBACK = _env_bool('ML_LIVE_EV_GATED_MARKET', PROFILE.ev_gated_market_fallback)
USE_CONFIDENCE_SIZING = _env_bool('ML_LIVE_CONFIDENCE_SIZING', True)
CONFIDENCE_LOW_PROB = _env_float('ML_LIVE_CONFIDENCE_LOW_PROB', 0.72)
CONFIDENCE_HIGH_PROB = _env_float('ML_LIVE_CONFIDENCE_HIGH_PROB', 0.78)
CONFIDENCE_LOW_FRACTION = _env_float('ML_LIVE_CONFIDENCE_LOW_FRACTION', 0.15)
CONFIDENCE_MID_FRACTION = _env_float('ML_LIVE_CONFIDENCE_MID_FRACTION', POSITION_FRACTION)
CONFIDENCE_HIGH_FRACTION = _env_float('ML_LIVE_CONFIDENCE_HIGH_FRACTION', 0.35)
# When False, buy_thr is used as-is (breakeven math cannot raise the bar).
USE_DYNAMIC_THRESHOLD = _env_bool('ML_LIVE_USE_DYNAMIC_THRESHOLD', True)
RUN_FOREVER = _env_bool('ML_LIVE_RUN_FOREVER', False)
LOOP_INTERVAL_SEC = _env_int('ML_LIVE_LOOP_INTERVAL_SEC', 900)
# 0 = no wall-clock limit. Set to 24 to run for one day, then exit cleanly.
LOOP_MAX_RUNTIME_HOURS = _env_float('ML_LIVE_MAX_RUNTIME_HOURS', 0.0)
# One-off escape hatch for adding the remaining free margin to one specific
# already-open position.  Both symbol and original entry price must match, and
# the state file records the operation after its first successful fill.
ONE_TIME_FULL_MARGIN_SYMBOL = _env_str(
    'ML_LIVE_ONE_TIME_FULL_MARGIN_SYMBOL', '').strip().upper()
ONE_TIME_FULL_MARGIN_ENTRY_PRICE = _env_float(
    'ML_LIVE_ONE_TIME_FULL_MARGIN_ENTRY_PRICE', 0.0)
# Kraken can reject a literal 100% order because opening fees/reference-price
# movement count against free margin. 99% is the executable full-margin target.
ONE_TIME_FULL_MARGIN_UTILIZATION = min(1.0, max(0.0, _env_float(
    'ML_LIVE_ONE_TIME_FULL_MARGIN_UTILIZATION', 0.99)))


def confidence_size_multiplier(probability: float, threshold: float) -> float:
    """Return multiplier that maps signal confidence to target margin fraction."""
    if POSITION_FRACTION <= 0:
        return 0.0

    low_prob = max(CONFIDENCE_LOW_PROB, threshold)
    high_prob = max(CONFIDENCE_HIGH_PROB, low_prob)
    if probability < low_prob:
        target_fraction = CONFIDENCE_LOW_FRACTION
    elif probability < high_prob:
        target_fraction = CONFIDENCE_MID_FRACTION
    else:
        target_fraction = CONFIDENCE_HIGH_FRACTION
    return max(0.0, target_fraction / POSITION_FRACTION)


def one_time_full_margin_key() -> str:
    """Stable state key for the narrowly targeted one-time margin add-on."""
    if not ONE_TIME_FULL_MARGIN_SYMBOL or ONE_TIME_FULL_MARGIN_ENTRY_PRICE <= 0:
        return ''
    return (
        f"{ONE_TIME_FULL_MARGIN_SYMBOL}@"
        f"{ONE_TIME_FULL_MARGIN_ENTRY_PRICE:.8f}"
    )


def one_time_full_margin_pending(state: dict, symbol: str, position: dict) -> bool:
    """Only match the configured symbol and the original position's fill."""
    key = one_time_full_margin_key()
    if not key or symbol.upper() != ONE_TIME_FULL_MARGIN_SYMBOL:
        return False
    if key in state.get('consumed_one_time_margin_overrides', []):
        return False
    entry_price = float(position.get('entry_price', 0.0) or 0.0)
    # The alert rounds XRP to four decimals. Accommodate sub-tick fill detail
    # while preventing a later XRP position from accidentally matching.
    tolerance = max(0.00005, ONE_TIME_FULL_MARGIN_ENTRY_PRICE * 0.00005)
    return abs(entry_price - ONE_TIME_FULL_MARGIN_ENTRY_PRICE) <= tolerance


def consume_one_time_full_margin(state: dict) -> None:
    """Persist successful use so later scheduled runs cannot repeat it."""
    key = one_time_full_margin_key()
    if not key:
        return
    consumed = state.setdefault('consumed_one_time_margin_overrides', [])
    if key not in consumed:
        consumed.append(key)


def build_live_strategy() -> MLSwingStrategy:
    """Build the strategy with the exact environment settings used live."""
    cost_model = build_cost_model(
        PROFILE,
        maker_entry_fee=MAKER_ENTRY_FEE,
        taker_entry_fee=TAKER_ENTRY_FEE,
        taker_exit_fee=TAKER_EXIT_FEE,
        margin_open_fee=MARGIN_OPEN_FEE,
        margin_rollover_fee_4h=ROLLOVER_FEE_4H,
        spread_buffer=SPREAD_BUFFER,
        slippage_buffer=SLIPPAGE_BUFFER,
        minimum_edge=MINIMUM_EDGE,
    )
    return create_ml_strategy(
        PROFILE,
        cost_model,
        horizon=HORIZON,
        buy_thr=BUY_THR,
        sell_thr=SELL_THR,
        exit_thr=EXIT_THR,
        use_fng_features=USE_FNG_FEATURES,
        use_fng_filter=USE_FNG_FILTER,
        long_only=LONG_ONLY,
        use_cost_aware_labels=USE_COST_AWARE_LABELS,
        use_btc_features=USE_BTC_FEATURES,
        use_btc_regime_filter=USE_BTC_REGIME_FILTER,
        use_relative_strength_filter=USE_RELATIVE_STRENGTH_FILTER,
        use_expected_value_filter=USE_EXPECTED_VALUE_FILTER,
        ev_cost_multiplier=EV_COST_MULTIPLIER,
        use_ev_exit=USE_EV_EXIT,
        use_dynamic_threshold=USE_DYNAMIC_THRESHOLD,
    )


def size_trade(
    usable_margin: float,
    position_fraction: float,
    conf_mult: float,
    regime_mult: float,
    leverage: int,
    price: float,
    min_volume: float,
    allow_min_bump: bool = True,
) -> tuple[float, float, str]:
    """
    Pick margin and volume for one trade.

    Default: use position_fraction of usable margin.
    Small accounts: if that is below Kraken's min volume, bump up to the
    minimum as long as usable margin can cover it. Never exceed usable margin.
    Returns (margin_usd, volume, note).
    """
    target_margin = usable_margin * position_fraction * conf_mult * regime_mult
    volume = (target_margin * leverage) / price if price > 0 else 0.0
    note = 'target'

    if volume >= min_volume and target_margin > 0:
        return target_margin, volume, note

    # Target size is too small for Kraken — try the exchange minimum.
    min_margin = (min_volume * price) / leverage if leverage > 0 else float('inf')
    if allow_min_bump and min_margin <= usable_margin and min_margin > 0:
        return min_margin, min_volume, 'bumped_to_exchange_min'

    return target_margin, volume, 'below_min'


def load_state() -> dict:
    """Read our record of open ML positions, or start empty."""
    path = Path(STATE_FILE)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {'open': {}, 'closed': []}


def save_state(state: dict) -> None:
    state['updated_at'] = pd.Timestamp.utcnow().isoformat()
    Path(STATE_FILE).write_text(json.dumps(state, indent=2, default=str))


def pair_map(config: Config) -> dict:
    """Map yfinance symbol -> its Kraken pair + minimum order volume."""
    return {p.yf_symbol: p for p in config.TRADING_PAIRS}


def normalize_kraken_pair(pair: str | None) -> str:
    """Normalize Kraken pair aliases enough for open-position comparisons."""
    if not pair:
        return ''
    value = ''.join(ch for ch in str(pair).upper() if ch.isalnum())
    aliases = {
        'XXBTZUSD': 'XBTUSD',
        'XBTZUSD': 'XBTUSD',
        'XETHZUSD': 'ETHUSD',
        'XXRPZUSD': 'XRPUSD',
        'XRPZUSD': 'XRPUSD',
        'XXDGZUSD': 'XDGUSD',
        'XDGZUSD': 'XDGUSD',
    }
    value = aliases.get(value, value)
    if value.endswith('ZUSD'):
        value = value[:-4] + 'USD'
    if value.startswith('XETH'):
        value = 'ETH' + value[4:]
    return value


def live_open_pair_names(kraken: KrakenClient) -> set[str]:
    """Return normalized Kraken pair names that currently have live exposure."""
    open_positions = kraken.get_open_positions()
    pairs = set()
    for pos_id, pos_data in open_positions.items():
        for raw_pair in (pos_data.get('pair'), pos_id):
            normalized = normalize_kraken_pair(raw_pair)
            if normalized:
                pairs.add(normalized)
    return pairs


def pair_is_live_open(pair: str, open_pairs: set[str]) -> bool:
    """Check whether a Kraken pair is present in a normalized live-position set."""
    return normalize_kraken_pair(pair) in open_pairs


def enter_position(kraken: KrakenClient, kraken_pair: str, order_type: str,
                   volume: float, leverage: int, fallback_price: float,
                   allow_market_fallback: bool) -> tuple:
    """
    Open a position, trying the cheap way first.

    1. Place a post-only LIMIT order at the best bid (buy) / best ask (sell),
       which pays the lower maker fee if it fills.
    2. Wait up to MAKER_WAIT_SEC, checking every few seconds.
    3. If it hasn't fully filled, cancel it and MARKET-order the remainder,
       so we always end up with the full position this run.

    Returns (average_fill_price, how, filled_volume).
    """
    if not USE_MAKER_ENTRY:
        kraken.place_order(pair=kraken_pair, order_type=order_type,
                           volume=volume, leverage=leverage, reduce_only=False)
        return fallback_price, 'taker', volume

    # Rest the order on our side of the spread so it can't cross (= maker).
    try:
        bid, ask = kraken.get_bid_ask(kraken_pair)
        decimals = kraken.get_pair_decimals(kraken_pair)
        limit_price = round(bid if order_type == 'buy' else ask, decimals)
        result = kraken.place_order(pair=kraken_pair, order_type=order_type,
                                    volume=volume, leverage=leverage, reduce_only=False,
                                    ordertype='limit', price=limit_price, post_only=True)
        txid = result.get('txid', [None])[0]
    except Exception as e:
        print(f"   maker entry failed ({e})")
        if not allow_market_fallback:
            return None, 'maker_failed_skip_taker', 0.0
        kraken.place_order(pair=kraken_pair, order_type=order_type,
                           volume=volume, leverage=leverage, reduce_only=False)
        return fallback_price, 'taker', volume

    if txid is None:
        return limit_price, 'maker_unknown', volume

    # Poll until filled or out of patience.
    deadline = time.time() + MAKER_WAIT_SEC
    while time.time() < deadline:
        time.sleep(5)
        try:
            info = kraken.query_order(txid)
        except Exception:
            continue
        if info.get('status') == 'closed':
            return float(info.get('price', limit_price) or limit_price), 'maker', volume

    # Not (fully) filled in time: cancel and market-order whatever is missing.
    filled_vol = 0.0
    try:
        kraken.cancel_order(txid)
        info = kraken.query_order(txid)
        filled_vol = float(info.get('vol_exec', 0) or 0)
    except Exception as e:
        print(f"   ⚠️ cancel/query after maker wait failed: {e}")

    remaining = volume - filled_vol
    if remaining > 0:
        if not allow_market_fallback:
            if filled_vol > 0:
                return limit_price, 'partial_maker_skip_taker', filled_vol
            return None, 'skipped_unfilled_maker', 0.0
        kraken.place_order(pair=kraken_pair, order_type=order_type,
                           volume=remaining, leverage=leverage, reduce_only=False)
    how = 'mixed' if filled_vol > 0 else 'taker'
    return fallback_price, how, volume


def main() -> None:
    config = Config()
    kraken = KrakenClient(config.KRAKEN_API_KEY, config.KRAKEN_API_SECRET, config.KRAKEN_API_URL)
    telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    strategy = build_live_strategy()
    cost_model = strategy.cost_model
    pairs = pair_map(config)

    mode = "🧪 DRY-RUN (no real orders)" if DRY_RUN else "💰 REAL MONEY"
    print(f"ML LIVE TRADER | {mode} | strategy={STRATEGY_VERSION.upper()} | "
          f"coins={SYMBOLS} | {POSITION_FRACTION:.0%}/trade @ {LEVERAGE}x")

    if not config.KRAKEN_API_KEY or not config.KRAKEN_API_SECRET:
        print("❌ Missing Kraken credentials - aborting.")
        return

    state = load_state()
    actions = []

    # Fresh data for active symbols plus any legacy open positions we still
    # need to manage after a symbol-set change.
    data_by_symbol = {}
    data_symbols = list(dict.fromkeys(SYMBOLS + list(state['open'].keys())))
    for symbol in data_symbols:
        try:
            df = get_history(symbol, PERIOD, INTERVAL)
            if not df.empty:
                data_by_symbol[symbol] = df
        except Exception as e:
            print(f"  ⚠️ {symbol}: data error {e}")
    btc_data = None
    btc_regimes = None
    if USE_BTC_FEATURES or USE_BTC_REGIME_FILTER or USE_RELATIVE_STRENGTH_FILTER:
        try:
            btc_data = get_history('BTC-USD', PERIOD, INTERVAL)
            btc_regimes = compute_btc_regime_frame(btc_data)
        except Exception as e:
            print(f"❌ Could not read BTC regime data: {e}")
            return

    # Read live account state (safe, read-only).
    try:
        available_margin = kraken.get_available_margin()
    except Exception as e:
        print(f"❌ Could not read margin: {e}")
        return
    usable_margin = available_margin / MARGIN_SAFETY_FACTOR
    try:
        exchange_open_pairs = set() if DRY_RUN else live_open_pair_names(kraken)
    except Exception as e:
        print(f"❌ Could not read live open positions: {e}")
        return

    # ── 1) Close positions: time limit OR model says bail early ──
    for symbol in list(state['open'].keys()):
        pos = state['open'][symbol]
        df = data_by_symbol.get(symbol)
        if df is None:
            continue
        now = df.index[-1]
        time_due = now >= pd.Timestamp(pos['exit_due'])
        early_exit = False
        exit_prob = pos.get('prob_up', 0.5)
        if not time_due and pos['direction'] == 'long' and EXIT_THR > 0:
            early_exit, exit_prob = strategy.should_exit_early(df, btc_data)
        btc_exit = False
        if USE_BTC_REGIME_FILTER and btc_regimes is not None:
            btc_state = btc_regime_state(btc_regimes, now)
            btc_exit = btc_state.block_new_entries
        if btc_exit:
            early_exit = True
        if not time_due and not early_exit:
            continue

        current_price = float(df['Close'].iloc[-1])
        kp = pairs.get(symbol)
        if kp is None:
            actions.append(f"⚠️ close {symbol} skipped: no Kraken pair mapping")
            continue
        try:
            if not DRY_RUN:
                if not pair_is_live_open(kp.kraken_pair, exchange_open_pairs):
                    state['closed'].append({**pos, 'symbol': symbol, 'exit_price': current_price,
                                            'exit_time': str(now), 'pnl_pct': 0.0,
                                            'exit_reason': 'missing_on_exchange',
                                            'exit_prob_up': round(exit_prob, 3)})
                    del state['open'][symbol]
                    actions.append(
                        f"STATE CLEANUP {symbol}: tracked position was not open on Kraken; "
                        "no close order sent")
                    continue
                kraken.close_position(kp.kraken_pair, pos['direction'], pos['volume'], LEVERAGE)
                exchange_open_pairs.discard(normalize_kraken_pair(kp.kraken_pair))
            # Informational realized P&L (price move x leverage).
            if pos['direction'] == 'long':
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100 * LEVERAGE
            else:
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100 * LEVERAGE
            reason = 'btc_regime' if btc_exit else 'model_exit' if early_exit else 'time'
            state['closed'].append({**pos, 'symbol': symbol, 'exit_price': current_price,
                                    'exit_time': str(now), 'pnl_pct': round(pnl_pct, 3),
                                    'exit_reason': reason, 'exit_prob_up': round(exit_prob, 3)})
            del state['open'][symbol]
            tag = f" ({reason}, p_up={exit_prob:.2f})" if early_exit else ""
            actions.append(f"CLOSE {symbol} {pos['direction']} @ {current_price:.4f} -> {pnl_pct:+.2f}%{tag}")
        except Exception as e:
            actions.append(f"⚠️ close {symbol} failed: {e}")

    # A narrowly targeted, stateful one-time add-on for an already-open trade.
    # This is intentionally separate from normal 33% entry sizing: it only
    # matches the configured original fill and consumes itself after a fill.
    margin_committed_this_run = 0.0
    topup_symbol = ONE_TIME_FULL_MARGIN_SYMBOL
    topup_pos = state['open'].get(topup_symbol) if topup_symbol else None
    if topup_pos and one_time_full_margin_pending(state, topup_symbol, topup_pos):
        df = data_by_symbol.get(topup_symbol)
        kp = pairs.get(topup_symbol)
        if df is None or kp is None:
            actions.append(f"skip one-time full-margin {topup_symbol}: missing market data/pair")
        elif topup_pos.get('direction') != 'long':
            actions.append(f"skip one-time full-margin {topup_symbol}: position is not long")
        else:
            sig = strategy.get_signal(df, btc_data)
            if sig.signal != 'BUY':
                actions.append(
                    f"skip one-time full-margin {topup_symbol}: BUY prediction no longer active "
                    f"(p={sig.prob_up:.2f}, thr={sig.dynamic_threshold:.2f})"
                )
            else:
                current_price = float(df['Close'].iloc[-1])
                margin_usd, volume, _ = size_trade(
                    usable_margin=usable_margin,
                    position_fraction=ONE_TIME_FULL_MARGIN_UTILIZATION,
                    conf_mult=1.0,
                    regime_mult=1.0,
                    leverage=LEVERAGE,
                    price=current_price,
                    min_volume=kp.min_volume,
                    allow_min_bump=False,
                )
                taker_cost = cost_model.estimated_total_cost(HORIZON, 'taker')
                taker_ev = expected_value(sig.prob_up, sig.avg_win, sig.avg_loss, taker_cost)
                allow_market_fallback = (
                    taker_ev > 0 and taker_ev > EV_COST_MULTIPLIER * taker_cost
                    if EV_GATED_MARKET_FALLBACK else True
                )
                try:
                    entry_price, fill_how, filled_volume = enter_position(
                        kraken, kp.kraken_pair, 'buy', volume, LEVERAGE,
                        current_price, allow_market_fallback)
                    if entry_price is None or filled_volume <= 0:
                        actions.append(
                            f"skip one-time full-margin {topup_symbol}: order did not fill"
                        )
                    else:
                        old_volume = float(topup_pos.get('volume', 0.0) or 0.0)
                        old_entry = float(topup_pos.get('entry_price', entry_price) or entry_price)
                        total_volume = old_volume + filled_volume
                        weighted_entry = (
                            (old_entry * old_volume + entry_price * filled_volume) / total_volume
                            if total_volume > 0 else entry_price
                        )
                        fill_fraction = min(1.0, filled_volume / volume) if volume > 0 else 0.0
                        allocated_margin = margin_usd * fill_fraction
                        topup_pos['entry_price'] = weighted_entry
                        topup_pos['volume'] = round(total_volume, 8)
                        topup_pos['margin_usd'] = round(
                            float(topup_pos.get('margin_usd', 0.0) or 0.0) + allocated_margin,
                            2,
                        )
                        topup_pos.setdefault('topups', []).append({
                            'time': str(df.index[-1]),
                            'price': entry_price,
                            'volume': round(filled_volume, 8),
                            'margin_usd': round(allocated_margin, 2),
                            'fill': fill_how,
                            'reason': 'one_time_full_margin',
                        })
                        consume_one_time_full_margin(state)
                        margin_committed_this_run = allocated_margin
                        actions.append(
                            f"ONE-TIME FULL-MARGIN {topup_symbol} ADD @ {entry_price:.4f} "
                            f"vol={filled_volume:.6f} ({fill_how}, margin ${allocated_margin:.2f})"
                        )
                except Exception as e:
                    actions.append(f"⚠️ one-time full-margin {topup_symbol} failed: {e}")

    # ── 2) Rank eligible high-margin signals, then open top opportunities ──
    latest_ts = max(df.index[-1] for df in data_by_symbol.values()) if data_by_symbol else pd.Timestamp.utcnow()
    btc_state = btc_regime_state(btc_regimes, latest_ts) if btc_regimes is not None else None
    allowed_max_open = min(MAX_OPEN, btc_state.max_positions) if USE_BTC_REGIME_FILTER and btc_state else MAX_OPEN
    open_slots = max(0, allowed_max_open - len(state['open']))
    if USE_BTC_REGIME_FILTER and btc_state and btc_state.block_new_entries:
        actions.append(f"skip all entries: BTC regime weak ({','.join(btc_state.reasons) or 'weak'})")

    candidates = []
    for symbol in SYMBOLS:
        if symbol in state['open']:
            continue
        df = data_by_symbol.get(symbol)
        kp = pairs.get(symbol)
        if df is None or kp is None:
            continue
        if not DRY_RUN and pair_is_live_open(kp.kraken_pair, exchange_open_pairs):
            actions.append(
                f"skip {symbol}: live Kraken position already open outside bot state")
            continue

        sig = strategy.get_signal(df, btc_data)
        if sig.signal not in ('BUY', 'SELL'):
            if sig.blocked_reason == 'fng_fear_bucket':
                actions.append(f"skip {symbol}: F&G fear bucket 25-40 (p_up={sig.prob_up:.2f})")
            elif sig.blocked_reason:
                actions.append(
                    f"skip {symbol}: {sig.blocked_reason} "
                    f"(p={sig.prob_up:.2f}, ev={sig.expected_value:.3%}, "
                    f"thr={sig.dynamic_threshold:.2f}, rs7={sig.relative_strength_7d:.2%})")
            continue
        if LONG_ONLY and sig.signal == 'SELL':
            actions.append(f"skip {symbol}: SELL signal ignored (long-only mode, p_up={sig.prob_up:.2f})")
            continue

        candidates.append((symbol, kp, df, sig))

    candidates.sort(key=lambda item: item[3].score, reverse=True)

    # Prefer coins we can actually afford on a small account (min-size bump).
    def can_afford(item) -> bool:
        symbol, kp, df, sig = item
        price = float(df['Close'].iloc[-1])
        min_margin = (kp.min_volume * price) / LEVERAGE if LEVERAGE > 0 else float('inf')
        return min_margin <= usable_margin

    candidates.sort(key=lambda item: (0 if can_afford(item) else 1, -item[3].score))

    remaining_margin = max(0.0, usable_margin - margin_committed_this_run)
    for symbol, kp, df, sig in candidates[:open_slots]:
        current_price = float(df['Close'].iloc[-1])
        conf_mult = (confidence_size_multiplier(sig.prob_up, sig.dynamic_threshold)
                     if USE_CONFIDENCE_SIZING else 1.0)
        margin_usd, volume, size_note = size_trade(
            usable_margin=usable_margin,
            position_fraction=POSITION_FRACTION,
            conf_mult=conf_mult,
            regime_mult=sig.regime_size_multiplier,
            leverage=LEVERAGE,
            price=current_price,
            min_volume=kp.min_volume,
            allow_min_bump=ALLOW_MIN_SIZE_BUMP,
        )
        if margin_usd > remaining_margin:
            margin_usd = remaining_margin
            volume = (margin_usd * LEVERAGE) / current_price if current_price > 0 else 0.0
            size_note = 'capped_to_remaining_margin'

        if margin_usd <= 0 or volume < kp.min_volume:
            min_margin = (kp.min_volume * current_price) / LEVERAGE if LEVERAGE > 0 else 0.0
            actions.append(
                f"skip {symbol}: need >= ${min_margin:.2f} margin for min size "
                f"{kp.min_volume} (have ${remaining_margin:.2f} remaining, "
                f"target ${usable_margin * POSITION_FRACTION:.2f})"
            )
            continue
        if size_note == 'bumped_to_exchange_min':
            actions.append(
                f"size {symbol}: bumped to Kraken min "
                f"(margin ${margin_usd:.2f}, vol={volume})"
            )
        elif size_note == 'capped_to_remaining_margin':
            actions.append(
                f"size {symbol}: capped to remaining margin "
                f"(margin ${margin_usd:.2f}, vol={volume})"
            )

        order_type = 'buy' if sig.signal == 'BUY' else 'sell'
        direction = 'long' if sig.signal == 'BUY' else 'short'
        now = df.index[-1]
        taker_cost = cost_model.estimated_total_cost(HORIZON, 'taker')
        taker_ev = expected_value(sig.prob_up, sig.avg_win, sig.avg_loss, taker_cost)
        allow_market_fallback = (
            taker_ev > 0 and taker_ev > EV_COST_MULTIPLIER * taker_cost
            if EV_GATED_MARKET_FALLBACK else True
        )
        try:
            fill_how = 'dry-run'
            entry_price = current_price
            filled_volume = volume
            if not DRY_RUN:
                entry_price, fill_how, filled_volume = enter_position(
                    kraken, kp.kraken_pair, order_type, volume, LEVERAGE,
                    current_price, allow_market_fallback)
            if entry_price is None or filled_volume <= 0:
                actions.append(
                    f"skip {symbol}: maker not filled and taker EV too low "
                    f"(maker_ev={sig.expected_value:.3%}, taker_ev={taker_ev:.3%})")
                continue
            state['open'][symbol] = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': str(now),
                'exit_due': str(now + timedelta(hours=HORIZON)),
                'volume': round(filled_volume, 8),
                'prob_up': round(sig.prob_up, 3),
                'expected_value': round(sig.expected_value, 5),
                'dynamic_threshold': round(sig.dynamic_threshold, 3),
                'estimated_cost': round(sig.estimated_cost, 5),
                'btc_regime': sig.btc_regime,
                'relative_strength_7d': round(sig.relative_strength_7d, 5),
                'margin_usd': round(margin_usd, 2),
                'leverage': LEVERAGE,
                'entry_fill': fill_how,
                'model_version': STRATEGY_VERSION,
            }
            actions.append(f"OPEN {symbol} {direction.upper()} @ {entry_price:.4f} "
                           f"vol={filled_volume:.6f} ({fill_how}, p={sig.prob_up:.2f}, "
                           f"thr={sig.dynamic_threshold:.2f}, ev={sig.expected_value:.2%}, "
                           f"score={sig.score:.2f}, margin ${margin_usd:.2f})")
            remaining_margin = max(0.0, remaining_margin - margin_usd)
            if not DRY_RUN:
                exchange_open_pairs.add(normalize_kraken_pair(kp.kraken_pair))
        except Exception as e:
            actions.append(f"⚠️ open {symbol} failed: {e}")

    save_state(state)

    # ── 3) Report ──
    n_closed = len(state['closed'])
    wins = [t for t in state['closed'] if t.get('pnl_pct', 0) > 0]
    win_rate = (len(wins) / n_closed * 100) if n_closed else 0.0

    header = f"{'🧪 ML DRY-RUN' if DRY_RUN else '💰 ML LIVE'} TRADER"
    body = (f"\nUsable margin: ${usable_margin:.2f}"
            f"\nOpen positions: {len(state['open'])}/{MAX_OPEN}"
            f"\nClosed trades: {n_closed} | Win rate: {win_rate:.1f}%")
    if actions:
        body += "\n\n<b>This run:</b>\n" + "\n".join(f"• {a}" for a in actions)
    else:
        body += "\n\n(no new actions this run)"

    print(body.replace('<b>', '').replace('</b>', ''))
    if actions:
        telegram.send(f"<b>{header}</b>{body}")


def run_forever() -> None:
    """Run decision cycles continuously for local/VPS 24/7 operation."""
    interval = max(60, LOOP_INTERVAL_SEC)
    started = time.time()
    max_runtime_sec = LOOP_MAX_RUNTIME_HOURS * 3600 if LOOP_MAX_RUNTIME_HOURS > 0 else None
    cycle = 0

    print(
        "ML LIVE LOOP | "
        f"interval={interval}s | "
        f"max_runtime_hours={LOOP_MAX_RUNTIME_HOURS:g}"
    )
    while True:
        cycle += 1
        print(f"\n── cycle {cycle} @ {pd.Timestamp.utcnow().isoformat()} ──")
        try:
            main()
        except KeyboardInterrupt:
            print("ML live loop interrupted; exiting.")
            return
        except Exception as e:
            print(f"❌ cycle {cycle} crashed: {e}")

        if max_runtime_sec is not None and time.time() - started >= max_runtime_sec:
            print("ML live loop reached runtime limit; exiting.")
            return

        sleep_for = interval
        if max_runtime_sec is not None:
            remaining = max_runtime_sec - (time.time() - started)
            if remaining <= 0:
                print("ML live loop reached runtime limit; exiting.")
                return
            sleep_for = min(sleep_for, remaining)
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("ML live loop interrupted; exiting.")
            return


if __name__ == "__main__":
    if RUN_FOREVER:
        run_forever()
    else:
        main()
