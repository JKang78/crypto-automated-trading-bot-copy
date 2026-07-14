"""
SHARED ML STRATEGY (single source of truth for backtest + live)

Two saved profiles — switch with ML_LIVE_STRATEGY=v2 or v3 (live/paper)
or ml_strategy_backtest.py --v3 for backtests.

V2 (default live): more trades, tuned for better temporal robustness.
  h=72, thr=0.70, F&G features + filter, adaptive exit at 0.40.
  Revalidate with ml_strategy_backtest.py --live-profile v2 before promotion.

V3: fewer trades, higher per-trade edge, cost-aware labels.
  h=72, thr=0.70, cost-aware labels, EV-filtered entries,
  EV-gated market entries, confidence sizing.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

from research_edge import build_features, build_labels, make_model

# Module-level cache so we only download Fear & Greed once per process.
_FNG_CACHE: Optional[pd.Series] = None


def fetch_fear_greed() -> pd.Series:
    """Download full daily Fear & Greed history (0=extreme fear, 100=greed)."""
    global _FNG_CACHE
    if _FNG_CACHE is not None:
        return _FNG_CACHE
    r = requests.get(
        'https://api.alternative.me/fng/',
        params={'limit': 0, 'format': 'json'},
        timeout=30,
    ).json()
    rows = [
        (pd.Timestamp(int(d['timestamp']), unit='s'), float(d['value']))
        for d in r['data']
    ]
    _FNG_CACHE = pd.Series(dict(rows)).sort_index()
    return _FNG_CACHE


def make_fng_features(fng: pd.Series) -> pd.DataFrame:
    """Daily F&G -> feature columns, shifted 1 day to avoid look-ahead."""
    df = pd.DataFrame(index=fng.index)
    df['fng'] = fng / 100.0
    df['fng_chg_7'] = fng.diff(7) / 100.0
    df['fng_vs_ma30'] = (fng - fng.rolling(30).mean()) / 100.0
    return df.shift(1)


@dataclass
class KrakenCostModel:
    """Expected all-in trading cost as a fraction of notional price return."""
    maker_entry_fee: float = 0.0023
    taker_entry_fee: float = 0.0040
    taker_exit_fee: float = 0.0040
    margin_open_fee: float = 0.0004
    margin_rollover_fee_4h: float = 0.0004
    spread_buffer: float = 0.0005
    slippage_buffer: float = 0.0010
    minimum_edge: float = 0.0075

    def estimated_total_cost(
        self,
        hold_hours: int,
        entry_order: str = 'maker',
        fee_multiplier: float = 1.0,
    ) -> float:
        """Return total expected round-trip cost as a fraction of notional."""
        entry_fee = self.taker_entry_fee if entry_order == 'taker' else self.maker_entry_fee
        rollovers = max(0, int(hold_hours) // 4)
        fee_cost = (
            entry_fee
            + self.taker_exit_fee
            + self.margin_open_fee
            + rollovers * self.margin_rollover_fee_4h
        )
        execution_cost = self.spread_buffer + self.slippage_buffer
        return fee_multiplier * fee_cost + execution_cost


@dataclass(frozen=True)
class StrategyProfile:
    """Named preset for V2 vs V3 — used by live, paper, and backtest runners."""
    version: str
    horizon: int
    buy_thr: float
    exit_thr: float
    use_cost_aware_labels: bool
    use_fng_features: bool
    use_fng_filter: bool
    long_only: bool
    margin_open_fee: float
    rollover_fee_4h: float
    minimum_edge: float
    use_ev_exit: bool
    ev_gated_market_fallback: bool
    use_confidence_sizing: bool
    use_btc_features: bool = False
    use_btc_regime_filter: bool = False
    use_relative_strength_filter: bool = False
    use_expected_value_filter: bool = False
    ev_cost_multiplier: float = 1.5


V2_PROFILE = StrategyProfile(
    version='v2',
    horizon=72,
    buy_thr=0.70,
    exit_thr=0.40,
    use_cost_aware_labels=False,
    use_fng_features=True,
    use_fng_filter=True,
    long_only=True,
    margin_open_fee=0.0002,
    rollover_fee_4h=0.0002,
    minimum_edge=0.0,
    use_ev_exit=False,
    ev_gated_market_fallback=False,
    use_confidence_sizing=False,
)

V3_PROFILE = StrategyProfile(
    version='v3',
    horizon=72,
    buy_thr=0.70,
    exit_thr=0.40,
    use_cost_aware_labels=True,
    use_fng_features=True,
    use_fng_filter=True,
    long_only=True,
    margin_open_fee=0.0004,
    rollover_fee_4h=0.0004,
    minimum_edge=0.0075,
    use_ev_exit=True,
    ev_gated_market_fallback=True,
    use_confidence_sizing=True,
    use_expected_value_filter=True,
)


def get_strategy_profile(version: str = 'v2') -> StrategyProfile:
    """Return V2 or V3 preset. Unknown values fall back to V2."""
    key = (version or 'v2').strip().lower()
    if key in ('v3', '3'):
        return V3_PROFILE
    return V2_PROFILE


def build_cost_model(profile: StrategyProfile, **overrides) -> KrakenCostModel:
    """Build KrakenCostModel from a profile, with optional field overrides."""
    fields = dict(
        margin_open_fee=profile.margin_open_fee,
        margin_rollover_fee_4h=profile.rollover_fee_4h,
        minimum_edge=profile.minimum_edge,
    )
    fields.update(overrides)
    return KrakenCostModel(**fields)


def create_ml_strategy(profile: StrategyProfile, cost_model: KrakenCostModel,
                       **overrides) -> 'MLSwingStrategy':
    """Instantiate MLSwingStrategy from a saved profile."""
    params = dict(
        horizon=profile.horizon,
        buy_thr=profile.buy_thr,
        sell_thr=0.0 if profile.long_only else 0.35,
        exit_thr=profile.exit_thr,
        use_fng_features=profile.use_fng_features,
        use_fng_filter=profile.use_fng_filter,
        long_only=profile.long_only,
        cost_model=cost_model,
        use_cost_aware_labels=profile.use_cost_aware_labels,
        use_btc_features=profile.use_btc_features,
        use_btc_regime_filter=profile.use_btc_regime_filter,
        use_relative_strength_filter=profile.use_relative_strength_filter,
        use_expected_value_filter=profile.use_expected_value_filter,
        ev_cost_multiplier=profile.ev_cost_multiplier,
        use_ev_exit=profile.use_ev_exit,
    )
    params.update(overrides)
    return MLSwingStrategy(**params)


def build_cost_aware_labels(
    data: pd.DataFrame,
    horizon: int,
    cost_model: KrakenCostModel,
    entry_order: str = 'maker',
    fee_multiplier: float = 1.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Label rows by whether the future move clears all expected costs plus a
    configurable minimum edge. Unknown future rows stay NaN.
    """
    future_return = data['Close'].shift(-horizon) / data['Close'] - 1
    required_return = (
        cost_model.estimated_total_cost(horizon, entry_order, fee_multiplier)
        + cost_model.minimum_edge
    )
    labels = pd.Series(np.where(future_return > required_return, 1.0, 0.0), index=data.index)
    labels[future_return.isna()] = np.nan
    return labels, future_return


def make_btc_relative_features(data: pd.DataFrame, btc_data: pd.DataFrame) -> pd.DataFrame:
    """Cross-asset features that compare the traded coin with BTC."""
    btc = btc_data.reindex(data.index, method='ffill')
    coin_close = data['Close']
    btc_close = btc['Close']
    df = pd.DataFrame(index=data.index)

    df['rel_btc_ret_24h'] = coin_close.pct_change(24) - btc_close.pct_change(24)
    df['rel_btc_ret_72h'] = coin_close.pct_change(72) - btc_close.pct_change(72)
    df['rel_btc_ret_7d'] = coin_close.pct_change(168) - btc_close.pct_change(168)

    coin_ema200 = coin_close.ewm(span=200, adjust=False).mean()
    btc_ema200 = btc_close.ewm(span=200, adjust=False).mean()
    df['coin_close_over_ema200'] = coin_close / coin_ema200
    df['btc_close_over_ema200'] = btc_close / btc_ema200
    return df


def relative_strength_7d(data: pd.DataFrame, btc_data: pd.DataFrame) -> pd.Series:
    """7-day return spread: coin return minus BTC return."""
    btc = btc_data.reindex(data.index, method='ffill')
    return data['Close'].pct_change(168) - btc['Close'].pct_change(168)


@dataclass
class BTCRegimeState:
    regime: str
    block_new_entries: bool
    max_positions: int
    size_multiplier: float
    reasons: list[str] = field(default_factory=list)


def compute_btc_regime_frame(btc_data: pd.DataFrame) -> pd.DataFrame:
    """Classify BTC market state from closed 1h candles."""
    close = btc_data['Close'].copy()
    idx = close.index

    close_4h = close.resample('4h').last().dropna()
    ema200_4h = close_4h.ewm(span=200, min_periods=50, adjust=False).mean()
    aligned_4h_close = close_4h.reindex(idx, method='ffill')
    aligned_4h_ema200 = ema200_4h.reindex(idx, method='ffill')

    ret_1h = close.pct_change(1)
    ret_24h = close.pct_change(24)
    realized_vol = close.pct_change().rolling(24).std()
    vol_threshold = realized_vol.rolling(180 * 24, min_periods=30 * 24).quantile(0.90)

    weak_trend = aligned_4h_close < aligned_4h_ema200
    weak_return = ret_24h < -0.03
    crash_1h = ret_1h < -0.05
    high_vol = realized_vol >= vol_threshold

    weak = (weak_trend | weak_return | crash_1h | high_vol).fillna(False)
    strong = (~weak & (aligned_4h_close > aligned_4h_ema200) & (ret_24h > 0)).fillna(False)

    regime = pd.Series('neutral', index=idx, dtype='object')
    regime[weak] = 'weak'
    regime[strong] = 'strong'

    return pd.DataFrame({
        'regime': regime,
        'btc_4h_close': aligned_4h_close,
        'btc_4h_ema200': aligned_4h_ema200,
        'btc_ret_1h': ret_1h,
        'btc_ret_24h': ret_24h,
        'btc_realized_vol_24h': realized_vol,
        'btc_vol_p90_180d': vol_threshold,
        'weak_trend': weak_trend.fillna(False),
        'weak_return': weak_return.fillna(False),
        'crash_1h': crash_1h.fillna(False),
        'high_vol': high_vol.fillna(False),
    })


def btc_regime_state(regime_frame: pd.DataFrame, ts: pd.Timestamp) -> BTCRegimeState:
    """Return the BTC regime state visible at timestamp ts."""
    if regime_frame.empty:
        return BTCRegimeState('neutral', False, 3, 0.5, ['btc_regime_unavailable'])

    pos = regime_frame.index.searchsorted(pd.Timestamp(ts), side='right') - 1
    if pos < 0:
        return BTCRegimeState('neutral', False, 3, 0.5, ['btc_regime_unavailable'])

    row = regime_frame.iloc[pos]
    reasons = [
        name for name in ('weak_trend', 'weak_return', 'crash_1h', 'high_vol')
        if bool(row.get(name, False))
    ]
    regime = str(row['regime'])
    if regime == 'weak':
        return BTCRegimeState('weak', True, 0, 0.0, reasons)
    if regime == 'strong':
        return BTCRegimeState('strong', False, 5, 1.0, reasons)
    return BTCRegimeState('neutral', False, 3, 0.5, reasons)


def compute_strict_btc_bear_frame(
    btc_data: pd.DataFrame,
    ema_span_days: int = 200,
    lookback_days: int = 56,
    min_drawdown: float = -0.10,
) -> pd.DataFrame:
    """
    Stricter bear classifier for research.

    Bear only when BOTH are true on daily closes:
      1. BTC is below its 200-day EMA
      2. BTC is down at least `min_drawdown` over `lookback_days` (default 8 weeks)

    This is meant to catch sustained bears (2018/2022), not every dip/high-vol day.
    """
    close = btc_data['Close'].copy()
    idx = close.index
    daily = close.resample('1D').last().dropna()
    ema200 = daily.ewm(span=ema_span_days, min_periods=max(50, ema_span_days // 2), adjust=False).mean()
    ret_lookback = daily.pct_change(lookback_days)

    below_ema = daily < ema200
    deep_down = ret_lookback <= min_drawdown
    bear = (below_ema & deep_down).fillna(False)

    aligned_daily = daily.reindex(idx, method='ffill')
    aligned_ema = ema200.reindex(idx, method='ffill')
    aligned_ret = ret_lookback.reindex(idx, method='ffill')
    aligned_bear = bear.reindex(idx, method='ffill').fillna(False)
    aligned_below = below_ema.reindex(idx, method='ffill').fillna(False)
    aligned_down = deep_down.reindex(idx, method='ffill').fillna(False)

    regime = pd.Series('non_bear', index=idx, dtype='object')
    regime[aligned_bear] = 'bear'

    return pd.DataFrame({
        'regime': regime,
        'btc_daily_close': aligned_daily,
        'btc_daily_ema200': aligned_ema,
        'btc_ret_8w': aligned_ret,
        'below_ema200': aligned_below,
        'deep_down_8w': aligned_down,
        'is_bear': aligned_bear,
    })


def strict_btc_bear_state(regime_frame: pd.DataFrame, ts: pd.Timestamp) -> BTCRegimeState:
    """Return strict bear/non-bear state at timestamp ts."""
    if regime_frame.empty:
        return BTCRegimeState('non_bear', False, 5, 1.0, ['strict_bear_unavailable'])

    pos = regime_frame.index.searchsorted(pd.Timestamp(ts), side='right') - 1
    if pos < 0:
        return BTCRegimeState('non_bear', False, 5, 1.0, ['strict_bear_unavailable'])

    row = regime_frame.iloc[pos]
    reasons = [
        name for name in ('below_ema200', 'deep_down_8w')
        if bool(row.get(name, False))
    ]
    if bool(row.get('is_bear', False)) or str(row.get('regime')) == 'bear':
        return BTCRegimeState('bear', True, 0, 0.0, reasons)
    return BTCRegimeState('non_bear', False, 5, 1.0, reasons)


def estimate_payoff_stats(future_return: pd.Series, trainable_mask: np.ndarray) -> tuple[float, float]:
    """Average gross winning return and average gross losing return magnitude."""
    known = np.asarray(future_return, dtype=float)[trainable_mask]
    known = known[~np.isnan(known)]
    if len(known) == 0:
        return 0.0, 0.0

    wins = known[known > 0]
    losses = known[known <= 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    return avg_win, avg_loss


def expected_value(probability: float, avg_win: float, avg_loss: float, estimated_cost: float) -> float:
    """Expected raw return after expected costs."""
    return probability * avg_win - (1.0 - probability) * avg_loss - estimated_cost


def dynamic_probability_threshold(
    base_threshold: float,
    avg_win: float,
    avg_loss: float,
    estimated_cost: float,
) -> float:
    """Raise the probability threshold when historical payoff/costs require it."""
    denom = avg_win + avg_loss
    if denom <= 0:
        return base_threshold
    breakeven = (avg_loss + estimated_cost) / denom
    return float(np.clip(max(base_threshold, breakeven), 0.50, 0.95))


def recent_volatility(data: pd.DataFrame, window: int = 72) -> float:
    ret = data['Close'].pct_change().tail(window)
    vol = float(ret.std()) if len(ret.dropna()) else 0.0
    return max(vol, 1e-6)


def build_enhanced_features(
    data: pd.DataFrame,
    use_fng: bool = True,
    btc_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Price features plus optional Fear & Greed, aligned to hourly bars."""
    feats = build_features(data)
    if btc_data is not None:
        feats = feats.join(make_btc_relative_features(data, btc_data))
    if not use_fng:
        return feats
    fng_feats = make_fng_features(fetch_fear_greed())
    return feats.join(fng_feats.reindex(data.index, method='ffill'))


def fng_at_time(ts: pd.Timestamp) -> float:
    """F&G level visible at time ts (shifted 1 day, no look-ahead)."""
    fng = fetch_fear_greed().shift(1)
    day = pd.Timestamp(ts).normalize()
    val = fng.get(day, np.nan)
    return float(val) if not np.isnan(val) else np.nan


def passes_fng_filter(ts: pd.Timestamp) -> bool:
    """
    Return False for the 25-40 fear bucket that historically loses money.
    If F&G data is unavailable, allow the trade (don't block on missing data).
    """
    val = fng_at_time(ts)
    if np.isnan(val):
        return True
    return not (25 <= val < 40)


@dataclass
class MLSignal:
    """The strategy's decision for one coin at the latest bar."""
    signal: Optional[str]   # 'BUY', 'SELL', or None
    confidence: float       # 0..1, how far the probability is from 0.5
    prob_up: float          # model probability that the V3 label is positive
    horizon: int            # intended hold length in bars
    blocked_reason: Optional[str] = None  # why a signal was suppressed
    expected_value: float = 0.0
    estimated_cost: float = 0.0
    dynamic_threshold: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    btc_regime: str = 'unknown'
    regime_size_multiplier: float = 1.0
    relative_strength_7d: float = 0.0
    score: float = 0.0


class MLSwingStrategy:
    """
    Longer-horizon ML entry model with V3 cost-aware enhancements.
    Call get_signal(data) for entries; should_exit_early(data) for open positions.
    """

    def __init__(
        self,
        horizon: int = 72,
        buy_thr: float = 0.70,
        sell_thr: float = 0.35,
        exit_thr: float = 0.40,
        model_name: str = 'logistic',
        min_train_rows: int = 1000,
        use_fng_features: bool = True,
        use_fng_filter: bool = True,
        long_only: bool = True,
        cost_model: Optional[KrakenCostModel] = None,
        use_cost_aware_labels: bool = False,
        use_btc_features: bool = False,
        use_btc_regime_filter: bool = False,
        use_relative_strength_filter: bool = False,
        use_expected_value_filter: bool = False,
        ev_cost_multiplier: float = 1.5,
        use_ev_exit: bool = True,
        use_dynamic_threshold: bool = True,
    ):
        self.horizon = horizon
        self.buy_thr = buy_thr
        self.sell_thr = sell_thr
        self.exit_thr = exit_thr
        self.model_name = model_name
        self.min_train_rows = min_train_rows
        self.use_fng_features = use_fng_features
        self.use_fng_filter = use_fng_filter
        self.long_only = long_only
        self.cost_model = cost_model or KrakenCostModel()
        self.use_cost_aware_labels = use_cost_aware_labels
        self.use_btc_features = use_btc_features
        self.use_btc_regime_filter = use_btc_regime_filter
        self.use_relative_strength_filter = use_relative_strength_filter
        self.use_expected_value_filter = use_expected_value_filter
        self.ev_cost_multiplier = ev_cost_multiplier
        self.use_ev_exit = use_ev_exit
        self.use_dynamic_threshold = use_dynamic_threshold

    def _train_and_predict(self, data: pd.DataFrame, btc_data: Optional[pd.DataFrame] = None) -> tuple:
        """
        Train on all past labelled rows and predict prob_up for the latest bar.
        Returns (prob_up, model, future_return, trainable_mask) or
        (None, None, None, None) if not enough data.
        """
        btc_for_features = btc_data if (self.use_btc_features and btc_data is not None) else None
        feats = build_enhanced_features(data, self.use_fng_features, btc_for_features)
        if self.use_cost_aware_labels:
            labels, future_return = build_cost_aware_labels(data, self.horizon, self.cost_model)
        else:
            labels, future_return = build_labels(data, self.horizon)

        X = feats.values
        y = labels.values
        valid_feat = ~np.isnan(X).any(axis=1)

        trainable = valid_feat & ~np.isnan(y)
        if trainable.sum() < self.min_train_rows or len(np.unique(y[trainable])) < 2:
            return None, None, None, None

        latest_idx = len(data) - 1
        if not valid_feat[latest_idx]:
            return None, None, None, None

        model = make_model(self.model_name)
        model.fit(X[trainable], y[trainable])
        prob_up = float(model.predict_proba(X[latest_idx:latest_idx + 1])[:, 1][0])
        return prob_up, model, future_return, trainable

    def get_signal(self, data: pd.DataFrame, btc_data: Optional[pd.DataFrame] = None) -> MLSignal:
        """Return a BUY/SELL/None decision for the latest bar."""
        prob_up, _model, future_return, trainable = self._train_and_predict(data, btc_data)
        if prob_up is None:
            return MLSignal(signal=None, confidence=0.0, prob_up=0.5,
                            horizon=self.horizon)

        estimated_cost = self.cost_model.estimated_total_cost(self.horizon, 'maker')
        avg_win, avg_loss = estimate_payoff_stats(future_return, trainable)
        if self.use_dynamic_threshold:
            dyn_thr = dynamic_probability_threshold(
                self.buy_thr, avg_win, avg_loss, estimated_cost)
        else:
            dyn_thr = float(self.buy_thr)
        ev = expected_value(prob_up, avg_win, avg_loss, estimated_cost)

        latest_ts = data.index[-1]
        regime = BTCRegimeState('unknown', False, 5, 1.0)
        if btc_data is not None:
            regime = btc_regime_state(compute_btc_regime_frame(btc_data), latest_ts)

        rs_7d = 0.0
        if btc_data is not None:
            rs = relative_strength_7d(data, btc_data)
            rs_7d = float(rs.iloc[-1]) if not np.isnan(rs.iloc[-1]) else 0.0

        blocked_reason = None
        if self.use_btc_regime_filter and regime.block_new_entries:
            blocked_reason = 'btc_regime_weak'
            signal = None
        elif self.use_relative_strength_filter and btc_data is not None and rs_7d <= 0:
            blocked_reason = 'relative_strength_7d_nonpositive'
            signal = None
        elif self.use_expected_value_filter and not (
            prob_up > dyn_thr and ev > 0 and ev > self.ev_cost_multiplier * estimated_cost
        ):
            blocked_reason = 'expected_value_too_low'
            signal = None
        elif prob_up > dyn_thr:
            signal = 'BUY'
        elif prob_up < self.sell_thr and not self.long_only:
            signal = 'SELL'
        else:
            signal = None

        if signal == 'BUY' and self.use_fng_filter:
            if not passes_fng_filter(latest_ts):
                blocked_reason = 'fng_fear_bucket'
                signal = None
        if signal == 'SELL' and self.long_only:
            blocked_reason = 'long_only'
            signal = None

        confidence = min(1.0, abs(prob_up - 0.5) * 2)
        score = ev / recent_volatility(data)
        return MLSignal(signal=signal, confidence=confidence, prob_up=prob_up,
                        horizon=self.horizon, blocked_reason=blocked_reason,
                        expected_value=ev, estimated_cost=estimated_cost,
                        dynamic_threshold=dyn_thr, avg_win=avg_win, avg_loss=avg_loss,
                        btc_regime=regime.regime,
                        regime_size_multiplier=regime.size_multiplier,
                        relative_strength_7d=rs_7d, score=score)

    def should_exit_early(self, data: pd.DataFrame, btc_data: Optional[pd.DataFrame] = None) -> tuple[bool, float]:
        """
        For an open long position: return (True, prob_up) if the model now
        thinks the net-profitable label is unlikely. Disabled when exit_thr=0.
        """
        if self.exit_thr <= 0:
            return False, 0.5
        prob_up, _model, future_return, trainable = self._train_and_predict(data, btc_data)
        if prob_up is None:
            return False, 0.5
        avg_win, avg_loss = estimate_payoff_stats(future_return, trainable)
        estimated_cost = self.cost_model.estimated_total_cost(self.horizon, 'maker')
        ev = expected_value(prob_up, avg_win, avg_loss, estimated_cost)
        if self.use_ev_exit:
            return (prob_up < self.exit_thr or ev < 0), prob_up
        return prob_up < self.exit_thr, prob_up
