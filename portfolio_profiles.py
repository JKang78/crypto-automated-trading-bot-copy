"""Named, reproducible portfolios shared by research and paper runners.

These profiles preserve each portfolio's per-coin threshold settings. The
research candidate uses those thresholds as dynamic floors; the current live
baseline uses them as fixed thresholds. Selecting a portfolio does not enable
live trading.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

from ml_strategy import (
    KrakenCostModel,
    MLSwingStrategy,
    StrategyProfile,
    V2_PROFILE,
    create_ml_strategy,
)


@dataclass(frozen=True)
class PortfolioProfile:
    """Immutable strategy selection and allocation settings for a portfolio."""

    name: str
    symbols: Mapping[str, StrategyProfile]
    position_fraction: float = 0.33
    leverage: int = 2
    max_open: int = 3
    use_dynamic_threshold: bool = False

    def __post_init__(self) -> None:
        # Copy first so callers cannot mutate the profile through their input map.
        object.__setattr__(self, "symbols", MappingProxyType(dict(self.symbols)))


SIX_COIN_V2 = PortfolioProfile(
    name="six_coin_v2",
    use_dynamic_threshold=True,
    symbols={
        "ADA-USD": replace(V2_PROFILE, horizon=24, buy_thr=0.70, exit_thr=0.40),
        "DOGE-USD": replace(V2_PROFILE, horizon=48, buy_thr=0.68, exit_thr=0.40),
        "LINK-USD": replace(V2_PROFILE, horizon=72, buy_thr=0.70, exit_thr=0.40),
        "SOL-USD": replace(V2_PROFILE, horizon=72, buy_thr=0.65, exit_thr=0.40),
        "XLM-USD": replace(V2_PROFILE, horizon=72, buy_thr=0.70, exit_thr=0.40),
        "XRP-USD": replace(V2_PROFILE, horizon=72, buy_thr=0.65, exit_thr=0.40),
    },
)

THREE_COIN_V2 = PortfolioProfile(
    name="three_coin_v2",
    symbols={
        symbol: replace(V2_PROFILE, horizon=24, buy_thr=0.70, exit_thr=0.40)
        for symbol in ("ADA-USD", "DOGE-USD", "SOL-USD")
    },
)

PORTFOLIO_PROFILES = MappingProxyType({
    profile.name: profile for profile in (SIX_COIN_V2, THREE_COIN_V2)
})


def get_portfolio_profile(name: str) -> PortfolioProfile:
    """Resolve a named portfolio, rejecting unknown names instead of falling back."""
    key = name.strip().lower() if isinstance(name, str) else ""
    try:
        return PORTFOLIO_PROFILES[key]
    except KeyError:
        choices = ", ".join(PORTFOLIO_PROFILES)
        raise ValueError(f"Unknown portfolio {name!r}; choose one of: {choices}") from None


def build_strategies(
    profile: PortfolioProfile, cost_model: KrakenCostModel,
) -> dict[str, MLSwingStrategy]:
    """Create independent strategies with the portfolio's threshold behavior."""
    return {
        symbol: create_ml_strategy(
            strategy_profile, cost_model,
            use_dynamic_threshold=profile.use_dynamic_threshold,
        )
        for symbol, strategy_profile in profile.symbols.items()
    }
