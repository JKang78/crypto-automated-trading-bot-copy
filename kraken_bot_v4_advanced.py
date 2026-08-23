"""
KRAKEN SWING BOT V4 - ADVANCED AI SYSTEM
Full integration of:
- V3: Multi-Asset + ML + Adaptive Regime + Correlation
- V4: Sentiment Analysis + On-Chain + Ensemble + RL Position Sizing
"""

import os
import time
import hmac
import hashlib
import base64
import binascii
import urllib.parse
from datetime import datetime, timedelta
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import json
import traceback
from pathlib import Path


def load_env_file(path: str = ".env") -> None:
    """Load KEY=value pairs from .env into os.environ."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            # .env file is the source of truth (fixes empty shell exports)
            os.environ[key] = value


load_env_file()


def env_csv_set(name: str, default: str, *, lower: bool = False) -> set:
    raw = os.getenv(name, default)
    values = set()
    for item in raw.split(','):
        value = item.strip()
        if not value:
            continue
        if value.lower() == 'all':
            return set()
        values.add(value.lower() if lower else value)
    return values

# ═══════════════════════════════════════════════════════════════════════════
#                    IMPORT V4 MODULES
# ═══════════════════════════════════════════════════════════════════════════

try:
    from sentiment_analyzer import (
        SentimentAnalyzer, 
        should_trade_based_on_sentiment
    )
    SENTIMENT_AVAILABLE = True
except ImportError:
    print("⚠️ sentiment_analyzer.py not found")
    SENTIMENT_AVAILABLE = False

try:
    from onchain_metrics import (
        OnChainAnalyzer,
        should_trade_based_on_onchain
    )
    ONCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ onchain_metrics.py not found")
    ONCHAIN_AVAILABLE = False

try:
    from ensemble_strategies import (
        EnsembleSystem,
        StrategyType
    )
    ENSEMBLE_AVAILABLE = True
except ImportError:
    print("⚠️ ensemble_strategies.py not found")
    ENSEMBLE_AVAILABLE = False

try:
    from rl_position_sizing import (
        RLPositionSizer,
        PositionSizeCalculator,
        MarketState
    )
    RL_AVAILABLE = True
except ImportError:
    print("⚠️ rl_position_sizing.py not found")
    RL_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#                          V4 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradingPair:
    yf_symbol: str
    kraken_pair: str
    min_volume: float
    allocation: float

class Config:
    # ══════════════════ APIs ══════════════════
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '').strip()
    KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET', '').strip()
    KRAKEN_API_URL = 'https://api.kraken.com'
    
    CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY', '')
    NEWSDATA_API_KEY = os.getenv('NEWSDATA_API_KEY', '')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # ══════════════════ V4 Features ══════════════════
    USE_SENTIMENT_ANALYSIS = os.getenv('USE_SENTIMENT_ANALYSIS', 'false').lower() == 'true'
    MIN_SENTIMENT_CONFIDENCE = float(os.getenv('MIN_SENTIMENT_CONFIDENCE', '0.5'))
    
    USE_ONCHAIN_ANALYSIS = os.getenv('USE_ONCHAIN_ANALYSIS', 'false').lower() == 'true'
    MIN_ONCHAIN_STRENGTH = float(os.getenv('MIN_ONCHAIN_STRENGTH', '0.5'))
    
    USE_ENSEMBLE_SYSTEM = os.getenv('USE_ENSEMBLE_SYSTEM', 'false').lower() == 'true'
    MIN_ENSEMBLE_CONSENSUS = float(os.getenv('MIN_ENSEMBLE_CONSENSUS', '0.6'))
    MIN_ENSEMBLE_CONFIDENCE = float(os.getenv('MIN_ENSEMBLE_CONFIDENCE', '0.6'))
    
    USE_RL_POSITION_SIZING = os.getenv('USE_RL_POSITION_SIZING', 'false').lower() == 'true'
    RL_LEARNING_RATE = float(os.getenv('RL_LEARNING_RATE', '0.1'))
    RL_DISCOUNT_FACTOR = float(os.getenv('RL_DISCOUNT_FACTOR', '0.95'))
    RL_EPSILON = float(os.getenv('RL_EPSILON', '0.1'))
    RL_STATE_FILE = os.getenv('RL_STATE_FILE', 'rl_state.json')
    
    # Ensemble weights
    WEIGHT_SWING = float(os.getenv('WEIGHT_SWING', '0.30'))
    WEIGHT_MOMENTUM = float(os.getenv('WEIGHT_MOMENTUM', '0.25'))
    WEIGHT_MEAN_REVERSION = float(os.getenv('WEIGHT_MEAN_REVERSION', '0.25'))
    WEIGHT_TREND_FOLLOWING = float(os.getenv('WEIGHT_TREND_FOLLOWING', '0.20'))
    
    # ══════════════════ Multi-Asset ══════════════════
    TRADING_PAIRS = [
        TradingPair('BTC-USD', 'XBTUSD', 0.0001, 0.30),
        TradingPair('ETH-USD', 'ETHUSD', 0.001, 0.25),
        TradingPair('ADA-USD', 'ADAUSD', 10.0, 0.25),
        TradingPair('SOL-USD', 'SOLUSD', 0.01, 0.20),
        TradingPair('XRP-USD', 'XRPUSD', 10.0, 0.25),
        # Added for the ML strategy (validated walk-forward, long-only).
        # Kraken minimums checked via the public AssetPairs API.
        TradingPair('LINK-USD', 'LINKUSD', 0.55, 0.20),
        TradingPair('DOGE-USD', 'XDGUSD', 50.0, 0.20),
        TradingPair('AVAX-USD', 'AVAXUSD', 0.5, 0.20),
        TradingPair('DOT-USD', 'DOTUSD', 3.9, 0.20),
        TradingPair('LTC-USD', 'LTCUSD', 0.1, 0.20),
        TradingPair('XLM-USD', 'XLMUSD', 30.0, 0.20),
    ]
    
    MAX_CORRELATION = float(os.getenv('MAX_CORRELATION', '0.7'))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '1'))
    
    # ══════════════════ Trading ══════════════════
    LEVERAGE = int(os.getenv('LEVERAGE', '3'))
    MIN_BALANCE = float(os.getenv('MIN_BALANCE', '1.0'))
    MARGIN_SAFETY_FACTOR = 1.5
    # Enter with post-only LIMIT (maker) orders to pay the lower maker fee.
    USE_MAKER_ORDERS = os.getenv('USE_MAKER_ORDERS', 'false').lower() == 'true'
    
    # ══════════════════ Risk ══════════════════
    BASE_STOP_LOSS = float(os.getenv('STOP_LOSS_PCT', '4.0'))
    BASE_TAKE_PROFIT = float(os.getenv('TAKE_PROFIT_PCT', '8.0'))
    BASE_TRAILING_STOP = float(os.getenv('TRAILING_STOP_PCT', '2.5'))
    MIN_PROFIT_FOR_TRAILING = float(os.getenv('MIN_PROFIT_FOR_TRAILING', '3.0'))
    V4_POSITION_STATE_FILE = os.getenv('V4_POSITION_STATE_FILE', 'v4_position_state.json')
    
    # ══════════════════ Strategy ══════════════════
    LOOKBACK_PERIOD = os.getenv('LOOKBACK_PERIOD', '180d')
    CANDLE_INTERVAL = os.getenv('CANDLE_INTERVAL', '1h')
    USE_VOLUME_FILTER = os.getenv('USE_VOLUME_FILTER', 'true').lower() == 'true'
    REGIME_LOOKBACK = int(os.getenv('REGIME_LOOKBACK', '30'))
    
    USE_ML_VALIDATION = os.getenv('USE_ML_VALIDATION', 'true').lower() == 'true'
    ML_CONFIDENCE_THRESHOLD = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.6'))

    # Bear-market V4 gate. Defaults intentionally target only BTC/ETH shorts:
    # the raw swing strategy over-trades, so standalone V4 must clear the same
    # kind of narrow direction/trend/cost filters used by the router.
    V4_ALLOWED_SYMBOLS = env_csv_set('V4_ALLOWED_SYMBOLS', 'BTC-USD,ETH-USD')
    V4_ALLOWED_DIRECTIONS = env_csv_set('V4_ALLOWED_DIRECTIONS', 'short', lower=True)
    V4_MIN_CONFIDENCE = float(os.getenv('V4_MIN_CONFIDENCE', '0.80'))
    V4_MAX_SIGNAL_AGE_HOURS = float(os.getenv('V4_MAX_SIGNAL_AGE_HOURS', '12'))
    V4_TREND_EMA = int(os.getenv('V4_TREND_EMA', '200'))
    V4_MIN_EXPECTANCY_PCT = float(os.getenv('V4_MIN_EXPECTANCY_PCT', '0.25'))
    V4_EXPECTED_HOLD_HOURS = int(os.getenv('V4_EXPECTED_HOLD_HOURS', '12'))
    V4_MAKER_ENTRY_FEE = float(os.getenv('V4_MAKER_ENTRY_FEE', '0.0040'))
    V4_TAKER_EXIT_FEE = float(os.getenv('V4_TAKER_EXIT_FEE', '0.0080'))
    V4_MARGIN_OPEN_FEE = float(os.getenv('V4_MARGIN_OPEN_FEE', '0.0004'))
    V4_ROLLOVER_FEE_4H = float(os.getenv('V4_ROLLOVER_FEE_4H', '0.0004'))
    V4_SPREAD_SLIPPAGE_BUFFER = float(os.getenv('V4_SPREAD_SLIPPAGE_BUFFER', '0.0015'))
    
    # ══════════════════ Mode ══════════════════
    DRY_RUN = os.getenv('DRY_RUN', 'true').lower() == 'true'


# ═══════════════════════════════════════════════════════════════════════════
#                    KRAKEN CLIENT (from V3)
# ═══════════════════════════════════════════════════════════════════════════

class KrakenClient:
    def __init__(self, api_key: str, api_secret: str, api_url: str):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.api_url = api_url
        self.session = requests.Session()
    
    def _decode_api_secret(self) -> bytes:
        """Decode Kraken base64 secret, fixing missing padding if needed."""
        secret = self.api_secret
        try:
            return base64.b64decode(secret, validate=True)
        except (binascii.Error, ValueError):
            padded = secret + '=' * (-len(secret) % 4)
            return base64.b64decode(padded)
    
    def _sign(self, urlpath: str, data: dict) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(self._decode_api_secret(), message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()
    
    def _request(self, endpoint: str, data: dict = None, private: bool = False) -> dict:
        url = self.api_url + endpoint
        
        if private:
            data = data or {}
            data['nonce'] = int(time.time() * 1000)
            headers = {
                'API-Key': self.api_key,
                'API-Sign': self._sign(endpoint, data)
            }
            response = self.session.post(url, data=data, headers=headers, timeout=30)
        else:
            response = self.session.get(url, params=data, timeout=30)
        
        response.raise_for_status()
        result = response.json()
        
        if result.get('error') and len(result['error']) > 0:
            raise Exception(f"Kraken error: {result['error']}")
        
        return result.get('result', {})
    
    def get_balance(self) -> Tuple[float, str]:
        result = self._request('/0/private/Balance', private=True)
        balances = {k: float(v) for k, v in result.items()}
        
        fiat = {'ZUSD': 'USD', 'USD': 'USD', 'ZEUR': 'EUR', 'EUR': 'EUR'}
        
        for key, currency in fiat.items():
            if key in balances and balances[key] > 0:
                return balances[key], currency
        
        return 0.0, 'EUR'
    
    def get_available_margin(self) -> float:
        try:
            result = self._request('/0/private/TradeBalance', private=True)
            margin_free = float(result.get('mf', 0))
            print(f"   💰 Available margin: {margin_free:.2f} EUR")
            return margin_free
        except Exception as e:
            print(f"   ⚠️ Error getting margin: {e}")
            balance, _ = self.get_balance()
            return balance * 0.5
    
    def get_open_positions(self) -> Dict:
        try:
            result = self._request('/0/private/OpenPositions', private=True)
            
            if not result:
                return {}
            
            consolidated = {}
            
            for pos_id, pos_data in result.items():
                vol = float(pos_data.get('vol', 0))
                vol_closed = float(pos_data.get('vol_closed', 0))
                open_vol = vol - vol_closed
                
                if open_vol <= 0:
                    continue
                
                pair = pos_data.get('pair', 'UNKNOWN')
                
                if pair in consolidated:
                    existing_vol = float(consolidated[pair].get('vol', 0))
                    consolidated[pair]['vol'] = str(existing_vol + open_vol)
                    
                    existing_cost = float(consolidated[pair].get('cost', 0))
                    new_cost = float(pos_data.get('cost', 0))
                    consolidated[pair]['cost'] = str(existing_cost + new_cost)
                else:
                    pos_data['vol'] = str(open_vol)
                    consolidated[pair] = pos_data
            
            return consolidated
            
        except Exception as e:
            if "No open positions" in str(e):
                return {}
            raise
    
    def place_order(self, pair: str, order_type: str, volume: float, 
                   leverage: int = None, reduce_only: bool = False,
                   ordertype: str = 'market', price: float = None,
                   post_only: bool = False) -> dict:
        # ordertype 'market' fills immediately (taker fee). 'limit' rests on the
        # order book at `price` and, if it fills, pays the cheaper maker fee.
        data = {
            'pair': pair,
            'type': order_type,
            'ordertype': ordertype,
            'volume': str(round(volume, 8))
        }

        # A limit order must say what price to rest at.
        if ordertype == 'limit' and price is not None:
            data['price'] = str(price)

        # 'post' = post-only: Kraken cancels the order rather than let it cross
        # the spread, guaranteeing we are the maker (never pay the taker fee).
        if post_only:
            data['oflags'] = 'post'
        
        if leverage and leverage > 1:
            data['leverage'] = str(leverage)
            if reduce_only:
                data['reduce_only'] = 'true'
        
        return self._request('/0/private/AddOrder', data=data, private=True)

    def get_open_orders(self) -> dict:
        """Return resting (still unfilled) orders, keyed by order id."""
        result = self._request('/0/private/OpenOrders', private=True)
        return result.get('open', {}) if result else {}

    def get_bid_ask(self, pair: str) -> Tuple[Optional[float], Optional[float]]:
        """Current best bid/ask from the public ticker (no auth needed)."""
        result = self._request('/0/public/Ticker', data={'pair': pair})
        for _, info in result.items():
            return float(info['b'][0]), float(info['a'][0])
        return None, None

    def cancel_order(self, txid: str) -> dict:
        """Cancel one resting order by its transaction id."""
        return self._request('/0/private/CancelOrder', data={'txid': txid}, private=True)

    def query_order(self, txid: str) -> dict:
        """
        Look up one order's current state. The result includes:
        - 'status': 'open', 'closed' (= fully filled), 'canceled', 'expired'
        - 'vol_exec': how much volume actually filled so far
        """
        result = self._request('/0/private/QueryOrders', data={'txid': txid}, private=True)
        return result.get(txid, {})

    def cancel_all_orders(self) -> dict:
        """
        Cancel every resting order. Called at the start of each run so a maker
        limit entry that did not fill last cycle cannot fill later at a price we
        no longer want.
        """
        return self._request('/0/private/CancelAll', private=True)

    def get_pair_decimals(self, pair: str) -> int:
        """
        Ask Kraken how many decimal places a pair's price allows. We must round
        limit prices to this precision or Kraken rejects the order.
        """
        try:
            result = self._request('/0/public/AssetPairs', data={'pair': pair})
            for _, info in result.items():
                return int(info.get('pair_decimals', 2))
        except Exception:
            pass
        return 2
    
    def close_position(self, pair: str, position_type: str, volume: float, 
                      leverage: int = None) -> dict:
        opposite_type = 'sell' if position_type == 'long' else 'buy'
        is_margin_position = leverage and leverage > 1
        
        if is_margin_position:
            return self.place_order(
                pair=pair,
                order_type=opposite_type,
                volume=volume,
                leverage=leverage,
                reduce_only=True
            )
        else:
            return self.place_order(
                pair=pair,
                order_type=opposite_type,
                volume=volume,
                leverage=None,
                reduce_only=False
            )


# ═══════════════════════════════════════════════════════════════════════════
#                    COMPONENTES V3 (Regime, ML, Correlation, Swing)
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    @staticmethod
    def detect(data: pd.DataFrame, lookback: int = 30) -> str:
        if len(data) < lookback:
            return 'RANGING'
        
        recent = data.tail(lookback)
        returns = recent['Close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_volatility = data['Close'].pct_change().dropna().std()
        
        high_low = (recent['High'] - recent['Low']).mean()
        close_change = abs(recent['Close'].iloc[-1] - recent['Close'].iloc[0])
        trend_strength = close_change / (high_low * lookback) if high_low > 0 else 0
        
        if volatility > avg_volatility * 1.5:
            return 'VOLATILE'
        elif trend_strength > 0.5:
            return 'TRENDING'
        else:
            return 'RANGING'
    
    @staticmethod
    def get_adapted_params(regime: str, base_sl: float, base_tp: float, 
                          base_trail: float) -> Dict[str, float]:
        adaptations = {
            'TRENDING': {
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 1.5,
                'trailing_stop_multiplier': 1.0,
            },
            'RANGING': {
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 0.7,
                'trailing_stop_multiplier': 0.8,
            },
            'VOLATILE': {
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 1.0,
                'trailing_stop_multiplier': 1.3,
            }
        }
        
        mult = adaptations.get(regime, adaptations['RANGING'])
        
        return {
            'stop_loss': base_sl * mult['stop_loss_multiplier'],
            'take_profit': base_tp * mult['take_profit_multiplier'],
            'trailing_stop': base_trail * mult['trailing_stop_multiplier']
        }


class MLSwingValidator:
    @staticmethod
    def calculate_features(data: pd.DataFrame, swing_idx: int) -> Dict[str, float]:
        if swing_idx < 20 or swing_idx >= len(data) - 5:
            return None
        
        window = data.iloc[swing_idx-20:swing_idx+5]
        features = {}
        
        avg_vol = window['Volume'].mean()
        swing_vol = data['Volume'].iloc[swing_idx]
        features['volume_ratio'] = swing_vol / avg_vol if avg_vol > 0 else 1.0
        
        returns = window['Close'].pct_change()
        features['momentum'] = returns.mean()
        features['volatility'] = returns.std()
        
        swing_price = data['Close'].iloc[swing_idx]
        recent_high = window['High'].max()
        recent_low = window['Low'].min()
        price_range = recent_high - recent_low
        features['price_position'] = (swing_price - recent_low) / price_range if price_range > 0 else 0.5
        
        sma_20 = window['Close'].mean()
        features['distance_from_sma'] = abs(swing_price - sma_20) / sma_20 if sma_20 > 0 else 0
        
        return features
    
    @staticmethod
    def validate_swing(data: pd.DataFrame, swing_idx: int, 
                      swing_type: str, threshold: float = 0.6) -> Tuple[bool, float]:
        features = MLSwingValidator.calculate_features(data, swing_idx)
        
        if features is None:
            return False, 0.0
        
        score = 0.0
        weights = 0.0
        
        if features['volume_ratio'] > 1.2:
            score += 0.3
        elif features['volume_ratio'] > 0.8:
            score += 0.15
        weights += 0.3
        
        if swing_type == 'LOW' and features['momentum'] < -0.001:
            score += 0.25
        elif swing_type == 'HIGH' and features['momentum'] > 0.001:
            score += 0.25
        elif abs(features['momentum']) < 0.0005:
            score += 0.125
        weights += 0.25
        
        if swing_type == 'LOW' and features['price_position'] < 0.3:
            score += 0.2
        elif swing_type == 'HIGH' and features['price_position'] > 0.7:
            score += 0.2
        weights += 0.2
        
        if features['distance_from_sma'] > 0.02:
            score += 0.15
        weights += 0.15
        
        if features['volatility'] > 0.01:
            score += 0.1
        weights += 0.1
        
        confidence = score / weights if weights > 0 else 0.0
        is_valid = confidence >= threshold
        
        return is_valid, confidence


class CorrelationManager:
    @staticmethod
    def calculate_correlation_matrix(data_dict: Dict[str, pd.DataFrame], 
                                     lookback: int = 30) -> pd.DataFrame:
        returns_dict = {}
        for symbol, data in data_dict.items():
            if len(data) >= lookback:
                returns = data['Close'].tail(lookback).pct_change().dropna()
                returns_dict[symbol] = returns
        
        if len(returns_dict) < 2:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_dict)
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    @staticmethod
    def check_position_correlation(open_positions: List[str], new_symbol: str,
                                   corr_matrix: pd.DataFrame, 
                                   max_corr: float = 0.7) -> Tuple[bool, float]:
        if corr_matrix.empty or new_symbol not in corr_matrix.columns:
            return True, 0.0
        
        max_correlation = 0.0
        
        for pos_symbol in open_positions:
            if pos_symbol in corr_matrix.columns and pos_symbol != new_symbol:
                corr = abs(corr_matrix.loc[new_symbol, pos_symbol])
                max_correlation = max(max_correlation, corr)
        
        can_open = max_correlation < max_corr
        
        return can_open, max_correlation


def calculate_volume_ma(data: pd.DataFrame, period: int = 20) -> pd.Series:
    return data['Volume'].rolling(window=period).mean()

class SwingDetectorV3:
    def __init__(self, data: pd.DataFrame, volume_filter: bool = True, 
                 use_ml: bool = True, ml_threshold: float = 0.6):
        self.data = data.copy()
        self.volume_filter = volume_filter
        self.use_ml = use_ml
        self.ml_threshold = ml_threshold
        self.volume_ma = calculate_volume_ma(data) if volume_filter else None
        
        self.st_highs = pd.Series(index=data.index, dtype=float)
        self.st_lows = pd.Series(index=data.index, dtype=float)
        self.int_highs = pd.Series(index=data.index, dtype=float)
        self.int_lows = pd.Series(index=data.index, dtype=float)
        self.ml_confidence = {}
    
    def _check_volume(self, i: int) -> bool:
        if not self.volume_filter or self.volume_ma is None:
            return True
        
        if pd.isna(self.volume_ma.iloc[i]):
            return True
        
        return self.data['Volume'].iloc[i] > self.volume_ma.iloc[i]
    
    def detect(self):
        highs = self.data['High'].values
        lows = self.data['Low'].values
        
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                if self._check_volume(i):
                    self.st_lows.iloc[i] = lows[i]
        
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                if self._check_volume(i):
                    self.st_highs.iloc[i] = highs[i]
        
        st_high_idx = self.st_highs.dropna().index.tolist()
        for i in range(1, len(st_high_idx) - 1):
            p, c, n = st_high_idx[i-1], st_high_idx[i], st_high_idx[i+1]
            
            if self.st_highs[c] > self.st_highs[p] and self.st_highs[c] > self.st_highs[n]:
                if self.use_ml:
                    idx = self.data.index.get_loc(c)
                    is_valid, confidence = MLSwingValidator.validate_swing(
                        self.data, idx, 'HIGH', self.ml_threshold
                    )
                    if is_valid:
                        self.int_highs[c] = self.st_highs[c]
                        self.ml_confidence[c] = confidence
                else:
                    self.int_highs[c] = self.st_highs[c]
        
        st_low_idx = self.st_lows.dropna().index.tolist()
        for i in range(1, len(st_low_idx) - 1):
            p, c, n = st_low_idx[i-1], st_low_idx[i], st_low_idx[i+1]
            
            if self.st_lows[c] < self.st_lows[p] and self.st_lows[c] < self.st_lows[n]:
                if self.use_ml:
                    idx = self.data.index.get_loc(c)
                    is_valid, confidence = MLSwingValidator.validate_swing(
                        self.data, idx, 'LOW', self.ml_threshold
                    )
                    if is_valid:
                        self.int_lows[c] = self.st_lows[c]
                        self.ml_confidence[c] = confidence
                else:
                    self.int_lows[c] = self.st_lows[c]
    
    def get_signal(self) -> Tuple[Optional[str], Optional[float], float]:
        self.detect()
        
        highs = self.int_highs.dropna()
        lows = self.int_lows.dropna()
        
        if len(highs) == 0 and len(lows) == 0:
            return None, None, 0.0
        
        last_high = highs.index[-1] if len(highs) > 0 else pd.Timestamp.min
        last_low = lows.index[-1] if len(lows) > 0 else pd.Timestamp.min
        
        if last_low > last_high:
            confidence = self.ml_confidence.get(last_low, 1.0)
            return 'BUY', lows.iloc[-1], confidence
        elif last_high > last_low:
            confidence = self.ml_confidence.get(last_high, 1.0)
            return 'SELL', highs.iloc[-1], confidence
        else:
            return None, None, 0.0


class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}" if token else None
    
    def send(self, message: str) -> bool:
        if not self.api_url or not self.chat_id:
            print(f"📱 {message}")
            return False
        
        try:
            if len(message) > 4000:
                message = message[:3900] + "\n..."
            
            data = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}
            response = requests.post(f"{self.api_url}/sendMessage", data=data, timeout=10)
            if not response.ok:
                # Retry without HTML if Telegram rejects formatting
                data = {'chat_id': self.chat_id, 'text': message}
                response = requests.post(f"{self.api_url}/sendMessage", data=data, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False


class PositionManagerV3:
    def __init__(self, config: Config, kraken: KrakenClient, telegram: Telegram):
        self.config = config
        self.kraken = kraken
        self.telegram = telegram
        self.state_file = Path(config.V4_POSITION_STATE_FILE)
        self.peak_prices = {}
        self.position_regimes = {}
        self.load_state()

    def load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            state = json.loads(self.state_file.read_text())
            self.peak_prices = {
                key: float(value)
                for key, value in state.get('peak_prices', {}).items()
                if value is not None
            }
        except Exception as e:
            print(f"   ⚠️ Could not load V4 position state: {e}")

    def save_state(self) -> None:
        try:
            state = {
                'peak_prices': self.peak_prices,
                'updated_at': datetime.utcnow().isoformat() + 'Z',
            }
            self.state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            print(f"   ⚠️ Could not save V4 position state: {e}")

    def sync_active_positions(self, active_position_ids: List[str]) -> None:
        active = set(active_position_ids)
        self.peak_prices = {
            key: value for key, value in self.peak_prices.items()
            if key in active
        }
        self.save_state()

    @staticmethod
    def normalize_position_type(pos_type: str) -> str:
        normalized = str(pos_type or '').lower()
        if normalized in ('buy', 'long'):
            return 'long'
        if normalized in ('sell', 'short'):
            return 'short'
        return normalized or 'unknown'

    @staticmethod
    def infer_leverage(pos_data: dict) -> float:
        raw = pos_data.get('leverage')
        if raw not in (None, '', 'none'):
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

        cost = float(pos_data.get('cost', 0) or 0)
        margin = float(pos_data.get('margin', 0) or 0)
        if margin > 0 and cost > 0:
            return max(1.0, round(cost / margin))
        return 1.0
    
    def check_position(self, pos_id: str, pos_data: dict, current_price: float,
                      regime_params: Dict[str, float]) -> Tuple[bool, str]:
        pos_type = self.normalize_position_type(pos_data.get('type', 'long'))
        entry_price = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        leverage = self.infer_leverage(pos_data)
        
        if pos_type == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
        
        stop_loss = regime_params['stop_loss']
        take_profit = regime_params['take_profit']
        trailing_stop = regime_params['trailing_stop']
        
        if pnl_pct <= -stop_loss:
            return True, f"🛑 Stop Loss: {pnl_pct:.2f}%"
        
        if pnl_pct >= take_profit:
            return True, f"🎯 Take Profit: {pnl_pct:.2f}%"
        
        if pnl_pct >= self.config.MIN_PROFIT_FOR_TRAILING:
            if pos_id not in self.peak_prices:
                self.peak_prices[pos_id] = current_price
            
            if pos_type == 'long' and current_price > self.peak_prices[pos_id]:
                self.peak_prices[pos_id] = current_price
            elif pos_type == 'short' and current_price < self.peak_prices[pos_id]:
                self.peak_prices[pos_id] = current_price
            
            peak = self.peak_prices[pos_id]
            if pos_type == 'long':
                peak_pnl = ((peak - entry_price) / entry_price) * 100 * leverage
            else:
                peak_pnl = ((entry_price - peak) / entry_price) * 100 * leverage
            
            drawdown = peak_pnl - pnl_pct
            
            if drawdown >= trailing_stop:
                return True, f"📉 Trailing: peak {peak_pnl:.2f}%, actual {pnl_pct:.2f}%"
        
        return False, ""
    
    def close_position(self, pair: str, pos_type: str, volume: float, 
                      reason: str, pos_data: dict, current_price: float):
        pos_type = self.normalize_position_type(pos_type)
        print(f"\n🔴 Closing {pair} ({pos_type})")
        print(f"   Reason: {reason}")
        
        leverage = int(self.infer_leverage(pos_data))
        
        if not self.config.DRY_RUN:
            try:
                result = self.kraken.close_position(
                    pair, pos_type, volume, leverage
                )
                print(f"   ✓ Closed: {result}")
                self.peak_prices.pop(pair, None)
                self.save_state()
            except Exception as e:
                print(f"   ❌ Error: {e}")
                return False
        else:
            print(f"   🧪 [SIMULATION]")
        
        entry = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        
        if pos_type == 'long':
            pnl_pct = ((current_price - entry) / entry) * 100 * leverage
        else:
            pnl_pct = ((entry - current_price) / entry) * 100 * leverage
        
        msg = f"""
🔴 <b>POSITION CLOSED</b>

<b>Pair:</b> {pair}
<b>Type:</b> {pos_type.upper()}
<b>Entry:</b> ${entry:.4f}
<b>Exit:</b> ${current_price:.4f}
<b>PnL:</b> {pnl_pct:+.2f}%
<b>Leverage:</b> {leverage}x
<b>Reason:</b> {reason}
"""
        if self.config.DRY_RUN:
            msg = "🧪 <b>SIMULATION</b>\n" + msg
        
        self.telegram.send(msg)
        return True


# ═══════════════════════════════════════════════════════════════════════════
#                    BOT V4 - ADVANCED AI SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class TradingBotV4:
    def __init__(self, config: Config):
        self.config = config
        self.kraken = KrakenClient(
            config.KRAKEN_API_KEY, 
            config.KRAKEN_API_SECRET, 
            config.KRAKEN_API_URL
        )
        self.telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.position_mgr = PositionManagerV3(config, self.kraken, self.telegram)
        
        # ═══════════════ INICIALIZAR COMPONENTES V4 ═══════════════
        
        # Sentiment Analyzer
        if config.USE_SENTIMENT_ANALYSIS and SENTIMENT_AVAILABLE:
            self.sentiment_analyzer = SentimentAnalyzer(
                cryptocompare_api_key=config.CRYPTOCOMPARE_API_KEY,
                newsdata_api_key=config.NEWSDATA_API_KEY
            )
            print("   ✓ Sentiment Analyzer enabled (Fear & Greed — no CryptoCompare needed)")
        else:
            self.sentiment_analyzer = None
        
        # On-Chain Analyzer
        if config.USE_ONCHAIN_ANALYSIS and ONCHAIN_AVAILABLE:
            self.onchain_analyzer = OnChainAnalyzer(config.CRYPTOCOMPARE_API_KEY)
            print("   ✓ On-Chain Analyzer enabled")
        else:
            self.onchain_analyzer = None
        
        # Ensemble System
        if config.USE_ENSEMBLE_SYSTEM and ENSEMBLE_AVAILABLE:
            weights = {
                StrategyType.SWING: config.WEIGHT_SWING,
                StrategyType.MOMENTUM: config.WEIGHT_MOMENTUM,
                StrategyType.MEAN_REVERSION: config.WEIGHT_MEAN_REVERSION,
                StrategyType.TREND_FOLLOWING: config.WEIGHT_TREND_FOLLOWING
            }
            self.ensemble = EnsembleSystem(weights=weights)
            print("   ✓ Ensemble System enabled")
        else:
            self.ensemble = None
        
        # RL Position Sizer
        if config.USE_RL_POSITION_SIZING and RL_AVAILABLE:
            self.rl_sizer = RLPositionSizer(
                learning_rate=config.RL_LEARNING_RATE,
                discount_factor=config.RL_DISCOUNT_FACTOR,
                epsilon=config.RL_EPSILON,
                state_file=config.RL_STATE_FILE
            )
            self.rl_calculator = PositionSizeCalculator(self.rl_sizer)
            print("   ✓ RL Position Sizing enabled")
            
            # Create empty file if it does not exist
            self._initialize_rl_state_file()
        else:
            self.rl_sizer = None
            self.rl_calculator = None
        
        # Trade history (for RL)
        self.trades_history = []
    
    def _initialize_rl_state_file(self):
        """Initialize RL state file if it does not exist."""
        try:
            if not os.path.exists(self.config.RL_STATE_FILE):
                with open(self.config.RL_STATE_FILE, 'w') as f:
                    json.dump({
                        'q_table': {}, 
                        'metadata': {
                            'created': datetime.now().isoformat(),
                            'num_states': 0
                        }
                    }, f)
                print(f"   📝 RL state file initialized: {self.config.RL_STATE_FILE}")
        except Exception as e:
            print(f"   ⚠️ Error initializing RL state: {e}")
    
    def get_market_data(self, symbol: str) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            period=self.config.LOOKBACK_PERIOD, 
            interval=self.config.CANDLE_INTERVAL
        )
        
        if data.empty:
            raise Exception(f"No data for {symbol}")

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        return data

    @staticmethod
    def signal_direction(signal: str) -> str:
        return 'long' if signal == 'BUY' else 'short'

    @staticmethod
    def latest_signal_time(detector: SwingDetectorV3, signal: str) -> Optional[pd.Timestamp]:
        points = detector.int_lows.dropna() if signal == 'BUY' else detector.int_highs.dropna()
        if points.empty:
            return None
        return pd.Timestamp(points.index[-1])

    def trend_filter(self, data: pd.DataFrame, direction: str) -> Tuple[bool, Dict]:
        ema_period = self.config.V4_TREND_EMA
        if ema_period <= 0:
            return True, {'trend_filter': 'disabled'}
        if len(data) < ema_period:
            return False, {
                'trend_filter': 'insufficient_data',
                'ema_period': ema_period,
                'bars': len(data),
            }

        close = data['Close']
        ema = float(close.ewm(span=ema_period, adjust=False).mean().iloc[-1])
        price = float(close.iloc[-1])
        allowed = price > ema if direction == 'long' else price < ema
        return allowed, {
            'trend_filter': 'ema',
            'ema_period': ema_period,
            'price': round(price, 8),
            'ema': round(ema, 8),
            'price_vs_ema_pct': round((price / ema - 1) * 100, 3) if ema else 0.0,
        }

    def estimate_after_cost_expectancy(self, data: pd.DataFrame, confidence: float) -> Dict:
        regime = RegimeDetector.detect(data, self.config.REGIME_LOOKBACK)
        params = RegimeDetector.get_adapted_params(
            regime,
            self.config.BASE_STOP_LOSS,
            self.config.BASE_TAKE_PROFIT,
            self.config.BASE_TRAILING_STOP,
        )
        leverage = max(1, int(self.config.LEVERAGE))
        stop_loss_pct = float(params['stop_loss'])
        take_profit_pct = float(params['take_profit'])
        win_probability = max(0.0, min(0.99, float(confidence)))
        rollovers = max(0, int(self.config.V4_EXPECTED_HOLD_HOURS) // 4)
        cost_fraction = (
            self.config.V4_MAKER_ENTRY_FEE
            + self.config.V4_TAKER_EXIT_FEE
            + self.config.V4_MARGIN_OPEN_FEE
            + rollovers * self.config.V4_ROLLOVER_FEE_4H
            + self.config.V4_SPREAD_SLIPPAGE_BUFFER
        )
        estimated_cost_pct = cost_fraction * 100 * leverage
        expected_gross_pct = (
            win_probability * take_profit_pct
            - (1.0 - win_probability) * stop_loss_pct
        )
        expected_net_pct = expected_gross_pct - estimated_cost_pct
        return {
            'regime': regime,
            'win_probability': round(win_probability, 4),
            'stop_loss_pct': round(stop_loss_pct, 4),
            'take_profit_pct': round(take_profit_pct, 4),
            'expected_hold_hours': self.config.V4_EXPECTED_HOLD_HOURS,
            'rollovers': rollovers,
            'estimated_cost_pct': round(estimated_cost_pct, 4),
            'expected_gross_pct': round(expected_gross_pct, 4),
            'expected_net_pct': round(expected_net_pct, 4),
            'minimum_expected_net_pct': self.config.V4_MIN_EXPECTANCY_PCT,
        }

    def evaluate_v4_entry_gate(
        self,
        pair: TradingPair,
        data: pd.DataFrame,
        detector: SwingDetectorV3,
        signal: str,
        confidence: float,
    ) -> Tuple[bool, Dict, List[str]]:
        direction = self.signal_direction(signal)
        details = {'direction': direction}
        reasons = []

        if self.config.V4_ALLOWED_SYMBOLS and pair.yf_symbol not in self.config.V4_ALLOWED_SYMBOLS:
            reasons.append(f"symbol_not_allowed:{pair.yf_symbol}")
        if self.config.V4_ALLOWED_DIRECTIONS and direction not in self.config.V4_ALLOWED_DIRECTIONS:
            reasons.append(f"direction_not_allowed:{direction}")

        signal_time = self.latest_signal_time(detector, signal)
        if signal_time is None:
            reasons.append('signal_time_unavailable')
        else:
            latest_time = pd.Timestamp(data.index[-1])
            age_hours = (latest_time - signal_time).total_seconds() / 3600
            details['signal_time'] = str(signal_time)
            details['signal_age_hours'] = round(age_hours, 2)
            if age_hours > self.config.V4_MAX_SIGNAL_AGE_HOURS:
                reasons.append(f"stale_signal:{age_hours:.1f}h")

        if confidence < self.config.V4_MIN_CONFIDENCE:
            reasons.append(
                f"confidence_below_min:{confidence:.3f}<{self.config.V4_MIN_CONFIDENCE:.3f}"
            )

        trend_ok, trend_details = self.trend_filter(data, direction)
        details.update(trend_details)
        if not trend_ok:
            reasons.append('trend_filter_reject')

        expectancy = self.estimate_after_cost_expectancy(data, confidence)
        details['expectancy'] = expectancy
        if expectancy['expected_net_pct'] < self.config.V4_MIN_EXPECTANCY_PCT:
            reasons.append(
                f"expectancy_below_min:{expectancy['expected_net_pct']:.3f}%"
                f"<{self.config.V4_MIN_EXPECTANCY_PCT:.3f}%"
            )

        return not reasons, details, reasons
    
    def analyze_trading_opportunity(self, 
                                   pair: TradingPair,
                                   data: pd.DataFrame,
                                   swing_signal: Tuple) -> Dict:
        """
        ═══════════════════════════════════════════════════════════════
        MULTI-LAYER V4 ANALYSIS
        ═══════════════════════════════════════════════════════════════
        """
        symbol = pair.yf_symbol
        signal, signal_price, swing_confidence = swing_signal
        
        print(f"\n🔍 Multi-Layer Analysis: {symbol}")
        print(f"   Swing Signal: {signal} (conf: {swing_confidence:.2f})")
        
        result = {
            'can_trade': False,
            'final_signal': None,
            'confidence': 0.0,
            'reasons': [],
            'capital': 0.0,
            'leverage': self.config.LEVERAGE,
            'v4_data': {}
        }
        
        # ═══════════════ LAYER 1: SENTIMENT ANALYSIS ═══════════════
        
        if self.sentiment_analyzer:
            print(f"\n   📊 Layer 1: Sentiment Analysis")
            try:
                sentiment = self.sentiment_analyzer.get_sentiment(symbol)
                
                if sentiment:
                    result['v4_data']['sentiment'] = {
                        'overall': sentiment.overall_score,
                        'news': sentiment.news_score,
                        'social': sentiment.social_score,
                        'signal_type': 'BULLISH' if sentiment.is_bullish() else ('BEARISH' if sentiment.is_bearish() else 'NEUTRAL')
                    }
                    
                    can_trade_sentiment = should_trade_based_on_sentiment(
                        sentiment,
                        signal,
                        min_confidence=self.config.MIN_SENTIMENT_CONFIDENCE
                    )
                    
                    if not can_trade_sentiment:
                        result['reasons'].append(
                            f"❌ Sentiment conflicts: {sentiment.overall_score:.2f}"
                        )
                        print(f"   ❌ Sentiment rejects: {signal}")
                        return result
                    
                    result['reasons'].append(
                        f"✓ Sentiment: {sentiment.overall_score:.2f} "
                        f"({result['v4_data']['sentiment']['signal_type']})"
                    )
                    print(f"   ✓ Sentiment confirms")
                else:
                    print(f"   ℹ️ Sentiment unavailable")
            except Exception as e:
                print(f"   ⚠️ Sentiment error: {e}")
        
        # ═══════════════ LAYER 2: ON-CHAIN METRICS ═══════════════
        
        if self.onchain_analyzer:
            print(f"\n   🔗 Layer 2: On-Chain Metrics")
            try:
                onchain = self.onchain_analyzer.get_onchain_signal(symbol)
                
                if onchain:
                    result['v4_data']['onchain'] = {
                        'signal_type': onchain.signal_type,
                        'strength': onchain.strength,
                        'metrics': onchain.metrics
                    }
                    
                    can_trade_onchain = should_trade_based_on_onchain(
                        onchain,
                        signal,
                        min_strength=self.config.MIN_ONCHAIN_STRENGTH
                    )
                    
                    if not can_trade_onchain:
                        result['reasons'].append(
                            f"❌ On-Chain conflicts: {onchain.signal_type}"
                        )
                        print(f"   ❌ On-Chain rejects: {signal}")
                        return result
                    
                    result['reasons'].append(
                        f"✓ On-Chain: {onchain.signal_type} (strength: {onchain.strength:.2f})"
                    )
                    print(f"   ✓ On-Chain confirms")
                else:
                    print(f"   ℹ️ On-Chain unavailable")
            except Exception as e:
                print(f"   ⚠️ On-chain error: {e}")
        
        # ═══════════════ LAYER 3: ENSEMBLE STRATEGIES ═══════════════
        
        if self.ensemble:
            print(f"\n   🎯 Layer 3: Ensemble Strategies")
            try:
                ensemble_decision = self.ensemble.get_ensemble_decision(
                    data, swing_signal
                )
                
                self.ensemble.print_decision_summary(ensemble_decision)
                
                result['v4_data']['ensemble'] = {
                    'final_signal': ensemble_decision.final_signal,
                    'confidence': ensemble_decision.confidence,
                    'consensus': ensemble_decision.consensus_level,
                    'votes': {str(k): str(v) for k, v in ensemble_decision.votes.items()}
                }
                
                # Check consensus and confidence
                if (ensemble_decision.final_signal != signal or
                    ensemble_decision.consensus_level < self.config.MIN_ENSEMBLE_CONSENSUS or
                    ensemble_decision.confidence < self.config.MIN_ENSEMBLE_CONFIDENCE):
                    
                    result['reasons'].append(
                        f"❌ Ensemble: {ensemble_decision.final_signal} "
                        f"(consensus: {ensemble_decision.consensus_level:.2f}, "
                        f"conf: {ensemble_decision.confidence:.2f})"
                    )
                    print(f"   ❌ Ensemble does not confirm")
                    return result
                
                result['reasons'].append(
                    f"✓ Ensemble: {ensemble_decision.final_signal} "
                    f"(consensus: {ensemble_decision.consensus_level:.2%}, "
                    f"conf: {ensemble_decision.confidence:.2%})"
                )
                result['confidence'] = ensemble_decision.confidence
                print(f"   ✓ Ensemble confirms with {ensemble_decision.consensus_level:.0%} consensus")
            except Exception as e:
                print(f"   ⚠️ Ensemble error: {e}")
                traceback.print_exc()
                result['confidence'] = swing_confidence
        else:
            result['confidence'] = swing_confidence
        
        # ═══════════════ FINAL DECISION ═══════════════
        
        result['can_trade'] = True
        result['final_signal'] = signal
        
        print(f"\n✅ DECISION: {signal}")
        print(f"   Final confidence: {result['confidence']:.2%}")
        
        return result
    
    def calculate_position_size(self, 
                               pair: TradingPair,
                               data: pd.DataFrame,
                               analysis_result: Dict,
                               available_margin: float) -> Tuple[float, int]:
        """
        ═══════════════════════════════════════════════════════════════
        LAYER 4: RL POSITION SIZING (or traditional)
        ═══════════════════════════════════════════════════════════════
        """
        
        if self.rl_calculator:
            print(f"\n   🤖 Layer 4: RL Position Sizing")
            try:
                # Get open positions
                positions = self.kraken.get_open_positions()
                open_positions_count = len(positions)
                
                # Calculate optimal size with RL
                capital, leverage = self.rl_calculator.get_optimal_size(
                    data=data,
                    signal_confidence=analysis_result['confidence'],
                    available_capital=available_margin,
                    base_leverage=self.config.LEVERAGE,
                    open_positions=open_positions_count,
                    recent_trades=self.trades_history[-20:],  # Last 20 trades
                    training=True  # Training mode
                )
                
                analysis_result['v4_data']['rl_sizing'] = {
                    'capital': capital,
                    'leverage': leverage,
                    'base_leverage': self.config.LEVERAGE
                }
                
                return capital, leverage
                
            except Exception as e:
                print(f"   ⚠️ RL sizing error: {e}")
                traceback.print_exc()
        
        # Fallback: traditional sizing
        print(f"\n   💰 Standard position sizing")
        allocation = pair.allocation
        capital = available_margin * allocation
        leverage = self.config.LEVERAGE
        
        return capital, leverage
    
    def open_position(self, 
                     pair: TradingPair,
                     analysis_result: Dict,
                     data: pd.DataFrame,
                     current_price: float,
                     capital: float,
                     leverage: int):
        """Open a position with full V4 analysis."""
        
        signal = analysis_result['final_signal']
        confidence = analysis_result['confidence']
        
        # Calculate volume
        volume = (capital * leverage) / current_price
        
        # Check minimum volume
        if volume < pair.min_volume:
            print(f"   ⚠️ Volume {volume:.8f} < minimum {pair.min_volume}")
            return
        
        try:
            print(f"\n🟢 Opening {signal} on {pair.yf_symbol}")
            print(f"   Price: ${current_price:.4f}")
            print(f"   Capital: ${capital:.2f}")
            print(f"   Leverage: {leverage}x")
            print(f"   Volume: {volume:.8f}")
            print(f"   Confidence: {confidence:.2%}")

            if signal == 'SELL' and leverage < 2:
                print("   ⚠️ Short entries require Kraken margin leverage >= 2x; skipping")
                return
            
            if not self.config.DRY_RUN:
                order_type = 'buy' if signal == 'BUY' else 'sell'

                if self.config.USE_MAKER_ORDERS:
                    # Maker (post-only limit) entry pays the lower fee. It may
                    # not fill; if so we simply re-decide next cycle (stale
                    # orders are cancelled at the start of run()).
                    decimals = self.kraken.get_pair_decimals(pair.kraken_pair)
                    limit_price = round(current_price, decimals)
                    result = self.kraken.place_order(
                        pair=pair.kraken_pair,
                        order_type=order_type,
                        volume=volume,
                        leverage=leverage,
                        reduce_only=False,
                        ordertype='limit',
                        price=limit_price,
                        post_only=True
                    )
                    print(f"   ✓ Maker limit order placed @ {limit_price}: {result}")
                else:
                    result = self.kraken.place_order(
                        pair=pair.kraken_pair,
                        order_type=order_type,
                        volume=volume,
                        leverage=leverage,
                        reduce_only=False
                    )
                    print(f"   ✓ Executed: {result}")
                
                # Save trade for RL
                trade_record = {
                    'symbol': pair.yf_symbol,
                    'entry_price': current_price,
                    'volume': volume,
                    'leverage': leverage,
                    'capital': capital,
                    'signal': signal,
                    'confidence': confidence,
                    'timestamp': datetime.now(),
                    'v4_data': analysis_result.get('v4_data', {}),
                    'closed': False
                }
                self.trades_history.append(trade_record)
                
            else:
                print(f"   🧪 [SIMULATION]")
            
            # V4 notification
            self._send_v4_notification(
                pair, signal, current_price, volume, leverage, 
                confidence, analysis_result['reasons'], analysis_result.get('v4_data', {})
            )
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Error: {error_msg}")
            self.telegram.send(f"❌ Error on {pair.yf_symbol}: {error_msg}")
    
    def _send_v4_notification(self, pair, signal, price, volume, 
                             leverage, confidence, reasons, v4_data):
        """Telegram notification with V4 details."""
        
        reasons_text = "\n".join([f"• {r}" for r in reasons])
        
        # Extract V4 data for display
        v4_summary = []
        
        if 'sentiment' in v4_data:
            sent = v4_data['sentiment']
            v4_summary.append(f"Sentiment: {sent['signal_type']} ({sent['overall']:.2f})")
        
        if 'onchain' in v4_data:
            onc = v4_data['onchain']
            v4_summary.append(f"On-Chain: {onc['signal_type']} ({onc['strength']:.2f})")
        
        if 'ensemble' in v4_data:
            ens = v4_data['ensemble']
            v4_summary.append(f"Ensemble: {ens['consensus']:.0%} consensus")
        
        if 'rl_sizing' in v4_data:
            rl = v4_data['rl_sizing']
            v4_summary.append(f"RL: ${rl['capital']:.2f} @ {rl['leverage']}x")
        
        v4_text = "\n".join(v4_summary) if v4_summary else "N/A"
        
        msg = f"""
🟢 <b>NEW POSITION V4</b>

<b>Pair:</b> {pair.yf_symbol} ({pair.kraken_pair})
<b>Signal:</b> {signal}
<b>Price:</b> ${price:.4f}
<b>Volume:</b> {volume:.8f}
<b>Leverage:</b> {leverage}x

<b>🤖 AI Analysis:</b>
<b>Confidence:</b> {confidence:.1%}
{v4_text}

<b>📊 Checks:</b>
{reasons_text}

<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        if self.config.DRY_RUN:
            msg = "🧪 <b>SIMULATION</b>\n" + msg
        
        self.telegram.send(msg)
    
    def run(self):
        """
        ═══════════════════════════════════════════════════════════════
        MAIN V4 LOOP
        ═══════════════════════════════════════════════════════════════
        """
        print("\n" + "="*70)
        print("KRAKEN TRADING BOT V4 - ADVANCED AI SYSTEM")
        print("="*70)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {'🧪 SIMULATION' if self.config.DRY_RUN else '💰 LIVE TRADING'}")
        print(f"ML Validation: {'✅' if self.config.USE_ML_VALIDATION else '❌'}")
        
        # Show active V4 features
        print("\n🤖 AI Features V4:")
        print(f"   Sentiment Analysis: {'✅' if self.sentiment_analyzer else '❌'}")
        print(f"   On-Chain Metrics: {'✅' if self.onchain_analyzer else '❌'}")
        print(f"   Ensemble System: {'✅' if self.ensemble else '❌'}")
        print(f"   RL Position Sizing: {'✅' if self.rl_calculator else '❌'}")
        print("="*70)
        
        try:
            # Get balance and margin
            balance, currency = self.kraken.get_balance()
            available_margin = self.kraken.get_available_margin()
            
            print(f"\n💰 Balance: {balance:.2f} {currency}")
            print(f"   Available margin: {available_margin:.2f} {currency}")
            
            if balance < self.config.MIN_BALANCE:
                print(f"⚠️ Insufficient balance (min: {self.config.MIN_BALANCE})")
                return
            
            usable_margin = available_margin / self.config.MARGIN_SAFETY_FACTOR
            print(f"   Usable margin: {usable_margin:.2f} {currency}")

            # Clear leftover maker (limit) entries from the previous cycle so
            # they cannot fill later at a stale price. Exits are never limit
            # orders, so this only cancels unfilled entries.
            if not self.config.DRY_RUN and self.config.USE_MAKER_ORDERS:
                try:
                    open_orders = self.kraken.get_open_orders()
                    if open_orders:
                        self.kraken.cancel_all_orders()
                        print(f"   🧹 Cancelled {len(open_orders)} stale order(s)")
                except Exception as e:
                    print(f"   ⚠️ Could not cancel stale orders: {e}")
            
            print("\n📊 Downloading multi-asset data...")
            market_data = {}
            for pair in self.config.TRADING_PAIRS:
                try:
                    data = self.get_market_data(pair.yf_symbol)
                    market_data[pair.yf_symbol] = data
                    print(f"   ✓ {pair.yf_symbol}: {len(data)} candles")
                except Exception as e:
                    print(f"   ❌ {pair.yf_symbol}: {e}")
            
            if not market_data:
                print("❌ Could not download market data")
                return
            
            # Calculate correlations
            print("\n🔗 Calculating correlations...")
            corr_matrix = CorrelationManager.calculate_correlation_matrix(
                market_data, lookback=30
            )
            
            if not corr_matrix.empty:
                print("   Correlation matrix:")
                print(corr_matrix.round(2))
            
            # Check open positions
            print("\n📊 Checking OPEN positions...")
            positions = self.kraken.get_open_positions()
            
            open_symbols = []
            active_position_ids = []
            total_margin_used = 0.0
            valid_position_count = len(positions)
            
            print(f"✅ {valid_position_count} active position(s)")
            
            if positions:
                for pair_key, pos_data in positions.items():
                    pos_margin = float(pos_data.get('margin', 0))
                    total_margin_used += pos_margin
                    
                    # Find trading pair
                    trading_pair = next(
                        (tp for tp in self.config.TRADING_PAIRS if tp.kraken_pair == pair_key),
                        None
                    )
                    
                    if not trading_pair or trading_pair.yf_symbol not in market_data:
                        continue
                    
                    open_symbols.append(trading_pair.yf_symbol)
                    data = market_data[trading_pair.yf_symbol]
                    current_price = float(data['Close'].iloc[-1])
                    
                    # Detect regime
                    regime = RegimeDetector.detect(data, self.config.REGIME_LOOKBACK)
                    regime_params = RegimeDetector.get_adapted_params(
                        regime, 
                        self.config.BASE_STOP_LOSS,
                        self.config.BASE_TAKE_PROFIT,
                        self.config.BASE_TRAILING_STOP
                    )
                    
                    print(f"\n   {trading_pair.yf_symbol} ({pair_key}) - Regime: {regime}")
                    print(f"   Margin used: {pos_margin:.2f} {currency}")
                    
                    # Check whether to close
                    normalized_pos_type = self.position_mgr.normalize_position_type(
                        pos_data.get('type', 'long')
                    )
                    should_close, reason = self.position_mgr.check_position(
                        pair_key, pos_data, current_price, regime_params
                    )
                    
                    if should_close:
                        volume = float(pos_data.get('vol', 0))
                        closed = self.position_mgr.close_position(
                            pair_key, normalized_pos_type, volume, reason, pos_data, current_price
                        )
                        if closed:
                            total_margin_used -= pos_margin
                            valid_position_count -= 1
                        else:
                            active_position_ids.append(pair_key)

                        # Update RL if active and the close was confirmed.
                        if closed and self.rl_sizer:
                            self._update_rl_on_close(
                                trading_pair.yf_symbol, 
                                pos_data, 
                                current_price, 
                                reason
                            )
                    else:
                        active_position_ids.append(pair_key)
                        print(f"   ✓ Hold position")
            else:
                print("✓ No open positions")

            self.position_mgr.sync_active_positions(active_position_ids)
            
            print(f"\n💰 Margin used: {total_margin_used:.2f} {currency}")
            print(f"   Remaining margin: {(available_margin - total_margin_used):.2f} {currency}")
            print(f"   Active positions: {valid_position_count}/{self.config.MAX_POSITIONS}")
            
            if valid_position_count >= self.config.MAX_POSITIONS:
                print(f"\nℹ️ Maximum positions reached ({self.config.MAX_POSITIONS})")
                return
            
            margin_for_new = (available_margin - total_margin_used) / self.config.MARGIN_SAFETY_FACTOR
            print(f"   Margin for new positions: {margin_for_new:.2f} {currency}")
            
            if margin_for_new < self.config.MIN_BALANCE * 0.5:
                print(f"⚠️ Insufficient margin for new positions")
                return
            
            print("\n🔍 Scanning for signals with V4 analysis...")
            
            validated_signals = []
            
            for pair in self.config.TRADING_PAIRS:
                if pair.yf_symbol not in market_data:
                    continue
                
                if pair.yf_symbol in open_symbols:
                    print(f"   ⭕ {pair.yf_symbol}: position already open")
                    continue
                
                data = market_data[pair.yf_symbol]
                current_price = float(data['Close'].iloc[-1])
                
                # Detect swing signal (V3)
                detector = SwingDetectorV3(
                    data, 
                    volume_filter=self.config.USE_VOLUME_FILTER,
                    use_ml=self.config.USE_ML_VALIDATION,
                    ml_threshold=self.config.ML_CONFIDENCE_THRESHOLD
                )
                signal, signal_price, confidence = detector.get_signal()
                
                if not signal:
                    print(f"   - {pair.yf_symbol}: no swing signal")
                    continue
                
                print(f"\n   🎯 {pair.yf_symbol}: {signal} signal detected")
                
                # ═══════════════════════════════════════════════════
                #       FULL V4 ANALYSIS
                # ═══════════════════════════════════════════════════
                
                analysis = self.analyze_trading_opportunity(
                    pair, data, (signal, signal_price, confidence)
                )
                
                if not analysis['can_trade']:
                    print(f"   ❌ Rejected by V4 analysis")
                    continue

                gate_ok, gate_details, gate_reasons = self.evaluate_v4_entry_gate(
                    pair,
                    data,
                    detector,
                    signal,
                    float(analysis.get('confidence', confidence) or 0.0)
                )
                analysis['v4_data']['entry_gate'] = gate_details
                if not gate_ok:
                    print(f"   ❌ Rejected by V4 entry gate: {', '.join(gate_reasons)}")
                    continue
                edge = gate_details['expectancy']['expected_net_pct']
                direction = gate_details['direction']
                analysis['reasons'].append(
                    f"✓ V4 entry gate: {direction}, est net edge {edge:+.2f}%"
                )
                
                # Check correlation
                can_open, max_corr = CorrelationManager.check_position_correlation(
                    open_symbols, pair.yf_symbol, corr_matrix, self.config.MAX_CORRELATION
                )
                
                if not can_open:
                    print(f"   ⚠️ Rejected by correlation ({max_corr:.2f})")
                    continue
                
                # Validated signal
                validated_signals.append({
                    'pair': pair,
                    'data': data,
                    'analysis': analysis,
                    'current_price': current_price
                })
                
                print(f"   ✅ {pair.yf_symbol} validated - Confidence: {analysis['confidence']:.2%}")
            
            if not validated_signals:
                print("\nℹ️ No valid signals after V4 analysis")
                return
            
            # Sort by confidence
            validated_signals.sort(key=lambda x: x['analysis']['confidence'], reverse=True)
            
            # Open positions
            positions_to_open = min(
                len(validated_signals), 
                self.config.MAX_POSITIONS - valid_position_count
            )
            
            print(f"\n🎯 Opening {positions_to_open} position(s) with AI V4...")
            
            remaining_margin = margin_for_new
            
            for sig in validated_signals[:positions_to_open]:
                # Calculate size BEFORE opening
                capital, leverage = self.calculate_position_size(
                    sig['pair'],
                    sig['data'],
                    sig['analysis'],
                    remaining_margin
                )
                
                # Open with pre-calculated values
                self.open_position(
                    sig['pair'],
                    sig['analysis'],
                    sig['data'],
                    sig['current_price'],
                    capital,
                    leverage
                )
                
                # Subtract margin used for next position
                margin_used = capital
                remaining_margin = max(0, remaining_margin - margin_used)
                
                if remaining_margin < self.config.MIN_BALANCE * 0.5:
                    print(f"   ⚠️ Insufficient remaining margin, stopping opens")
                    break
            
            # Always save (not only in LIVE mode)
            self.position_mgr.save_state()
            if self.rl_sizer:
                self.rl_sizer.save_state()
                print(f"\n💾 RL state saved")
            
            print("\n✅ Cycle completed")
            
        except Exception as e:
            msg = f"Error: {str(e)}"
            print(f"\n❌ {msg}")
            traceback.print_exc()
            self.telegram.send(f"❌ Bot V4 Error: {msg}")
            raise
    
    def _update_rl_on_close(self, symbol: str, pos_data: dict, 
                           exit_price: float, reason: str):
        """
        Update RL agent when a position is closed.
        """
        if not self.rl_sizer:
            return
        
        try:
            # Calculate PnL
            entry_price = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
            leverage = float(pos_data.get('leverage', 1))
            pos_type = pos_data.get('type', 'long')
            
            if pos_type == 'long':
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100 * leverage
            else:
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100 * leverage
            
            # Find trade in history
            matching_trades = [
                t for t in self.trades_history 
                if t['symbol'] == symbol and not t.get('closed', False)
            ]
            
            if not matching_trades:
                return
            
            trade = matching_trades[-1]  # Latest trade for this symbol

            # Mark as closed
            trade['closed'] = True
            trade['exit_price'] = exit_price
            trade['pnl_pct'] = pnl_pct
            trade['exit_reason'] = reason
            
            # Calculate reward
            trade_result = {
                'closed': True,
                'pnl_pct': pnl_pct,
                'exit_reason': reason
            }
            
            reward = self.rl_sizer.calculate_reward(trade_result)
            
            print(f"   🤖 RL: reward={reward:.3f} for PnL={pnl_pct:.2f}%")
            
        except Exception as e:
            print(f"   ⚠️ Error updating RL: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """V4 bot entry point."""
    
    print("\n" + "╔"*70)
    print("🚀 INITIALIZING KRAKEN TRADING BOT V4")
    print("╔"*70)
    
    config = Config()
    
    # Verify basic credentials
    if not config.KRAKEN_API_KEY or not config.KRAKEN_API_SECRET:
        print("❌ Missing Kraken credentials")
        return
    
    if len(config.KRAKEN_API_SECRET) < 50:
        print("❌ KRAKEN_API_SECRET looks too short or incomplete.")
        print("   Kraken private keys are usually ~88 characters (base64).")
        print("   Re-copy the FULL private key from Kraken → Settings → API.")
        print("   Paste into .env with no quotes and no spaces:")
        print("   KRAKEN_API_SECRET=paste_full_key_here")
        return
    
    # Verify V4 APIs
    missing_apis = []
    
    if config.USE_SENTIMENT_ANALYSIS and not config.CRYPTOCOMPARE_API_KEY and not config.NEWSDATA_API_KEY:
        print("\nℹ️ Sentiment uses free Fear & Greed Index (no CryptoCompare key needed)")
    
    if config.USE_ONCHAIN_ANALYSIS and not config.CRYPTOCOMPARE_API_KEY:
        missing_apis.append("CRYPTOCOMPARE_API_KEY (for On-Chain — or set USE_ONCHAIN_ANALYSIS=false)")
    
    if missing_apis:
        print("\n⚠️ WARNING: V4 features disabled due to missing APIs:")
        for api in missing_apis:
            print(f"   - {api}")
        print("\nThe bot will run without these features.")
    
    # Verify V4 modules
    missing_modules = []
    
    if config.USE_SENTIMENT_ANALYSIS and not SENTIMENT_AVAILABLE:
        missing_modules.append("sentiment_analyzer.py")
    
    if config.USE_ONCHAIN_ANALYSIS and not ONCHAIN_AVAILABLE:
        missing_modules.append("onchain_metrics.py")
    
    if config.USE_ENSEMBLE_SYSTEM and not ENSEMBLE_AVAILABLE:
        missing_modules.append("ensemble_strategies.py")
    
    if config.USE_RL_POSITION_SIZING and not RL_AVAILABLE:
        missing_modules.append("rl_position_sizing.py")
    
    if missing_modules:
        print("\n⚠️ WARNING: V4 modules not found:")
        for mod in missing_modules:
            print(f"   - {mod}")
        print("\nThe bot will run without these features.")
    
    print("\n" + "╔"*70)
    
    # Initialize and run bot
    bot = TradingBotV4(config)
    bot.run()
    
    print("\n" + "╔"*70)
    print("✅ BOT V4 EXECUTION COMPLETED")
    print("╔"*70)


if __name__ == "__main__":
    main()
