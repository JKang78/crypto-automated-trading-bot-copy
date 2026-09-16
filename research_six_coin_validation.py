"""Offline comparison of the six-coin research portfolio and three-coin baseline.

Default mode independently replays the allocation and costs of SAVED standalone
trade candidates. It does not claim to regenerate their ML signals. OHLC data is
used to expose losses while positions are open. No exchange or network access is
needed, and no live/paper account state is read or changed.

Optional --regenerate calls the shared walk-forward backtest, using only local
OHLC and an explicit daily Fear & Greed CSV (columns: date,value). This still
generates standalone candidates, not a joint portfolio signal simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from portfolio_profiles import PortfolioProfile, get_portfolio_profile


@dataclass(frozen=True)
class HistoricalCosts:
    """Explicit assumptions from the August report, not current verified fees."""

    name: str
    entry_fee: float
    exit_fee: float
    margin_open_fee: float = 0.0004
    rollover_fee_4h: float = 0.0004
    spread_slippage: float = 0.0015

    def total_rate(self, held_hours):
        return (
            self.entry_fee + self.exit_fee + self.margin_open_fee
            + np.floor(np.maximum(held_hours, 0) / 4) * self.rollover_fee_4h
            + self.spread_slippage
        )


COST_SCENARIOS = (
    HistoricalCosts("kraken_10k_high_margin", 0.0022, 0.0038),
    HistoricalCosts("kraken_0_volume_high_margin", 0.0040, 0.0080),
)
EXPECTED_HOLDOUT = {
    "six_coin_v2": {"return_pct": 148.9, "realized_drawdown_pct": 19.1, "trades": 31},
    "three_coin_v2": {"return_pct": 101.7, "realized_drawdown_pct": 7.9, "trades": 18},
}


def source_info(path: Path, frame: pd.DataFrame) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
    }


def load_local_history(symbol: str, cc_dir: Path, binance_dir: Path):
    base = symbol.split("-", 1)[0]
    paths = (cc_dir / f"{base}_1h.csv", binance_dir / f"{base}USDT_1h.csv")
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        raise ValueError(f"No local OHLC cache for {symbol}; expected one of {paths}")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError(f"Invalid or empty OHLC cache: {path}")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"OHLC timestamps must be sorted and unique: {path}")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError(f"OHLC contains missing/nonfinite values: {path}")
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"OHLC contains nonpositive prices: {path}")
    return frame, source_info(path, frame)


def saved_case_name(strategy) -> str:
    if strategy.horizon == 72 and strategy.buy_thr == 0.70:
        return "v2_baseline"
    return f"v2_h{strategy.horizon}_t{round(strategy.buy_thr * 100)}"


def load_candidates(profile: PortfolioProfile, directory: Path):
    frames, sources = [], {}
    for symbol, strategy in profile.symbols.items():
        case = saved_case_name(strategy)
        path = directory / f"{case}_trades.csv"
        if not path.is_file():
            raise ValueError(f"Missing saved candidates: {path}")
        frame = pd.read_csv(path)
        required = {
            "symbol", "direction", "entry_time", "exit_time", "entry_price",
            "exit_price", "bars_held", "score", "prob_up", "case",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing candidate columns in {path}: {required - set(frame)}")
        for column in ("entry_time", "exit_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_localize(None)
        selected = frame[frame.symbol == symbol].copy()
        if selected.empty:
            raise ValueError(f"No saved candidates for {symbol} in {path}")
        if not (selected.case == case).all() or not (selected.direction == "long").all():
            raise ValueError(f"Wrong strategy case/direction in {path}")
        numeric = selected[["entry_price", "exit_price", "bars_held", "score", "prob_up"]]
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite candidate values in {path}")
        hours = (selected.exit_time - selected.entry_time).dt.total_seconds() / 3600
        if not ((selected.bars_held > 0) & (hours >= selected.bars_held) & (selected.bars_held <= strategy.horizon)).all():
            raise ValueError(f"Candidate holding times inconsistent with hourly profile in {path}")
        if (selected[["entry_price", "exit_price"]] <= 0).any().any():
            raise ValueError(f"Invalid candidate price in {path}")
        frames.append(selected)
        sources[case] = source_info(path, frame.set_index("entry_time"))
    return pd.concat(frames, ignore_index=True), list(sources.values())


def regenerate_candidates(profile, histories, train_min, retrain_every):
    # Only imported in explicit regeneration mode; no call to any history downloader.
    from ml_strategy_backtest import backtest_symbol
    from ml_strategy import KrakenCostModel

    trades = []
    for symbol, strategy in profile.symbols.items():
        print(f"Regenerating {profile.name} {symbol}, {len(histories[symbol])} local bars", flush=True)
        cost_model = KrakenCostModel(
            maker_entry_fee=0.0023, taker_entry_fee=0.0040, taker_exit_fee=0.0040,
            margin_open_fee=strategy.margin_open_fee,
            margin_rollover_fee_4h=strategy.rollover_fee_4h,
            spread_buffer=0.0005, slippage_buffer=0.0010, minimum_edge=0.0,
        )
        result = backtest_symbol(
            symbol=symbol, data=histories[symbol], horizon=strategy.horizon,
            buy_thr=strategy.buy_thr, sell_thr=0.0, fee_rate=0.0023,
            leverage=profile.leverage, model_name="logistic", train_min=train_min,
            retrain_every=retrain_every, atr_stop_mult=0.0, atr_period=14,
            exit_thr=strategy.exit_thr, use_fng_features=strategy.use_fng_features,
            use_fng_filter=strategy.use_fng_filter,
            margin_open_fee=strategy.margin_open_fee, rollover_fee=strategy.rollover_fee_4h,
            cost_model=cost_model, entry_fee_rate=0.0023, exit_fee_rate=0.0040,
            use_dynamic_threshold=profile.use_dynamic_threshold,
        )
        trades.extend(result.get("trades", []))
    if not trades:
        raise ValueError(f"Regeneration returned no trades for {profile.name}")
    return pd.DataFrame(trades)


def max_drawdown(values) -> float:
    values = np.asarray(values, dtype=float)
    peaks = np.maximum.accumulate(values)
    return float(np.max(np.divide(peaks - values, peaks, out=np.zeros_like(values), where=peaks > 0)) * 100)


def allocate_candidates(candidates, profile, costs, starting_equity):
    """Reproduce realized-equity sizing while enforcing occupied margin and slots.

    Unrealized gains/losses do NOT change these historical allocation decisions.
    Mark-to-market risk is independently measured after allocation.
    """
    frame = candidates.sort_values(["entry_time", "score"], ascending=[True, False], kind="stable")
    entries = {timestamp: group for timestamp, group in frame.groupby("entry_time", sort=True)}
    event_times = sorted(set(entries) | set(frame.exit_time))
    equity, open_positions, selected = starting_equity, [], []
    realized_curve, peak_open, peak_margin_fraction = [equity], 0, 0.0
    rejected = {"position_cap": 0, "occupied_symbol": 0, "no_margin": 0}
    for timestamp in event_times:
        still_open = []
        for position in open_positions:
            if position["exit_time"] <= timestamp:
                equity += position["pnl_cash"]
                realized_curve.append(equity)
                selected.append(position)
            else:
                still_open.append(position)
        open_positions = still_open
        group = entries.get(timestamp)
        if group is None:
            continue
        free_margin = max(0.0, equity - sum(position["margin"] for position in open_positions))
        occupied = {position["symbol"] for position in open_positions}
        for row in group.to_dict("records"):
            if equity <= 0 or free_margin <= 0:
                rejected["no_margin"] += 1
                continue
            if len(open_positions) >= profile.max_open:
                rejected["position_cap"] += 1
                continue
            if row["symbol"] in occupied:
                rejected["occupied_symbol"] += 1
                continue
            margin = min(equity * profile.position_fraction, free_margin)
            gross = (row["exit_price"] / row["entry_price"] - 1) * profile.leverage
            cost = float(costs.total_rate(row["bars_held"])) * profile.leverage
            elapsed_hours = (row["exit_time"] - row["entry_time"]).total_seconds() / 3600
            elapsed_cost = float(costs.total_rate(elapsed_hours)) * profile.leverage
            position = {
                **row, "margin": margin, "cost_cash": margin * cost,
                "elapsed_hours": elapsed_hours,
                "extra_elapsed_rollover_cost_cash": margin * (elapsed_cost - cost),
                "net_return_pct": (gross - cost) * 100, "pnl_cash": margin * (gross - cost),
            }
            open_positions.append(position)
            occupied.add(row["symbol"])
            free_margin -= margin
            peak_open = max(peak_open, len(open_positions))
            peak_margin_fraction = max(peak_margin_fraction, sum(p["margin"] for p in open_positions) / equity)
    selected_frame = pd.DataFrame(selected)
    return selected_frame, {
        "trades": len(selected), "ending_equity": equity,
        "return_pct": (equity / starting_equity - 1) * 100,
        "realized_drawdown_pct": max_drawdown(realized_curve),
        "total_cost_cash": sum(position["cost_cash"] for position in selected),
        "extra_elapsed_rollover_cost_cash": sum(position["extra_elapsed_rollover_cost_cash"] for position in selected),
        "max_open": peak_open, "max_margin_fraction": peak_margin_fraction,
        "rejected_candidates": sum(rejected.values()), **{f"rejected_{key}": value for key, value in rejected.items()},
    }


def mark_to_market(selected, histories, costs, leverage, starting_equity):
    """Value accepted long positions hourly, net of accrued and estimated exit costs.

    The adverse-hour curve uses each open coin's own hourly low. Those lows need
    not occur simultaneously. It is a stress proxy, not an observed equity path.
    Entry is at candle close, so the entry candle's earlier low is excluded.
    """
    if selected.empty:
        return pd.DataFrame(), {"close_mtm_drawdown_pct": 0.0, "adverse_hour_drawdown_pct": 0.0, "minimum_close_mtm_equity": starting_equity, "missing_open_position_mark_hours": 0}
    index = pd.date_range(selected.entry_time.min(), selected.exit_time.max(), freq="h")
    realized_deltas = np.zeros(len(index))
    open_pnl, adverse_pnl = np.zeros(len(index)), np.zeros(len(index))
    missing_hours = 0
    for position in selected.to_dict("records"):
        start = index.get_loc(position["entry_time"])
        finish = index.get_loc(position["exit_time"])
        realized_deltas[finish] += position["pnl_cash"]
        market = histories[position["symbol"]].reindex(index[start:finish + 1])
        missing = market.Close.isna()
        if missing.iloc[0] or missing.iloc[-1]:
            raise ValueError(f"Missing entry/exit mark for {position['symbol']} trade {position['entry_time']}")
        missing_hours += int(missing.sum())
        # Missing bars cannot reveal the path; preserve last observed close and
        # disclose the number of imputed marks rather than silently claiming coverage.
        market["Close"] = market.Close.ffill()
        market.loc[missing, "Low"] = market.loc[missing, "Close"]
        if not np.isclose(market.Close.iloc[0], position["entry_price"], rtol=1e-7):
            raise ValueError(f"Entry price differs from OHLC cache for {position['symbol']} {position['entry_time']}")
        if position.get("exit_reason", "time") != "stop" and not np.isclose(market.Close.iloc[-1], position["exit_price"], rtol=1e-7):
            raise ValueError(f"Exit price differs from OHLC cache for {position['symbol']} {position['exit_time']}")
        held = np.arange(finish - start)
        close_return = market.Close.iloc[:-1].to_numpy() / position["entry_price"] - 1
        valuation_cost = costs.total_rate(held)
        open_pnl[start:finish] += position["margin"] * leverage * (close_return - valuation_cost)
        # At the entry close no earlier intrabar drawdown has been experienced.
        adverse_prices = market.Low.to_numpy(copy=True)
        adverse_prices[0] = position["entry_price"]
        adverse_return = adverse_prices / position["entry_price"] - 1
        # Include the exit candle's low before its close/realized PnL.
        adverse = position["margin"] * leverage * (adverse_return - costs.total_rate(np.arange(len(market))))
        adverse[-1] -= position["pnl_cash"]
        adverse_pnl[start:finish + 1] += adverse
    realized = starting_equity + np.cumsum(realized_deltas)
    closes, adverse = realized + open_pnl, realized + adverse_pnl
    # A candle's low precedes its close: compare it with previously observed
    # close peaks, never with a new peak first established at that same close.
    prior_peaks = np.maximum.accumulate(np.r_[starting_equity, closes])[:-1]
    adverse_dd = np.divide(prior_peaks - adverse, prior_peaks, out=np.zeros_like(adverse), where=prior_peaks > 0)
    close_drawdown = max_drawdown(np.r_[starting_equity, closes])
    curve = pd.DataFrame({"realized_equity": realized, "close_mtm_equity": closes, "adverse_hour_equity": adverse}, index=index)
    curve.index.name = "timestamp"
    return curve, {
        "close_mtm_drawdown_pct": close_drawdown,
        "adverse_hour_drawdown_pct": max(close_drawdown, float(max(0, adverse_dd.max()) * 100)),
        "minimum_close_mtm_equity": float(min(starting_equity, closes.min())),
        "missing_open_position_mark_hours": missing_hours,
    }


def write_report(directory, summary, manifest):
    lines = [
        "# Six-coin historical validation", "",
        f"Generated: {manifest['generated_at']}", "",
        f"Mode: **{manifest['mode']}**. Starting equity: {manifest['starting_equity']:.2f} account currency units.", "",
        "The fixed portfolios are compared without another optimizer search. Returns are cumulative, not annualized.", "",
        "| Portfolio | Period | Cost assumption | Trades | Return | Realized DD | Hourly-close DD | Adverse-hour DD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['profile']} | {row['period']} | {row['cost_scenario']} | {row['trades']} | "
            f"{row['return_pct']:+.2f}% | {row['realized_drawdown_pct']:.2f}% | "
            f"{row['close_mtm_drawdown_pct']:.2f}% | {row['adverse_hour_drawdown_pct']:.2f}% |"
        )
    lines.extend(["", "## Published holdout arithmetic", ""])
    for name, result in manifest["published_holdout_comparison"].items():
        lines.append(f"- {name}: {result}")
    lines.extend(["", "## Input coverage", ""])
    for symbol, info in manifest["ohlc_sources"].items():
        lines.append(f"- {symbol}: {info['start']} to {info['end']} ({info['rows']:,} hourly rows).")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in manifest["limitations"])
    lines.extend(["", "The manifest contains source hashes, profiles, dates, and all fee assumptions. Selected trades and equity paths are saved separately for inspection.", ""])
    (directory / "REPORT.md").write_text("\n".join(lines))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades-dir", type=Path, default=Path("research_outputs/nine_coin_optimizer"))
    parser.add_argument("--cc-dir", type=Path, default=Path("data_cc"))
    parser.add_argument("--binance-dir", type=Path, default=Path("data_binance"))
    parser.add_argument("--out-dir", type=Path, default=Path("validation_outputs/six_coin"))
    parser.add_argument("--starting-equity", type=float, default=170.0)
    parser.add_argument("--development-start", default="2021-01-01")
    parser.add_argument("--holdout-start", default="2024-01-01")
    parser.add_argument("--holdout-end", default="2027-01-01")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--fng-cache", type=Path, help="Local daily CSV with date,value columns; mandatory for regeneration")
    parser.add_argument("--train-min", type=int, default=4000)
    parser.add_argument("--retrain-every", type=int, default=720)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not np.isfinite(args.starting_equity) or args.starting_equity <= 0:
        raise ValueError("Starting equity must be positive and finite")
    dates = [pd.Timestamp(value) for value in (args.development_start, args.holdout_start, args.holdout_end)]
    if not dates[0] < dates[1] < dates[2]:
        raise ValueError("Require development-start < holdout-start < holdout-end")
    if args.train_min < 300 or args.retrain_every < 1:
        raise ValueError("train-min must be at least 300 and retrain-every must be positive")
    profiles = [get_portfolio_profile(name) for name in ("six_coin_v2", "three_coin_v2")]
    histories, ohlc_sources = {}, {}
    for symbol in profiles[0].symbols:
        histories[symbol], ohlc_sources[symbol] = load_local_history(symbol, args.cc_dir, args.binance_dir)
    fng_source = None
    if args.regenerate:
        if args.fng_cache is None or not args.fng_cache.is_file():
            raise ValueError("--regenerate needs --fng-cache pointing to a local daily date,value CSV; no download is attempted")
        import ml_strategy
        fng = pd.read_csv(args.fng_cache, parse_dates=["date"]).set_index("date")
        fng.index = pd.to_datetime(fng.index, utc=True).tz_localize(None)
        if "value" not in fng or fng.empty or not fng.index.is_unique or not fng.index.is_monotonic_increasing:
            raise ValueError("F&G cache requires sorted unique dates and value column")
        if not fng.value.between(0, 100).all():
            raise ValueError("F&G values must be in [0, 100]")
        if fng.index.max() < max(frame.index.max().normalize() for frame in histories.values()) - pd.Timedelta(days=1):
            raise ValueError("F&G cache does not cover the latest OHLC day")
        ml_strategy._FNG_CACHE = fng.value.astype(float)
        fng_source = source_info(args.fng_cache, fng)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    periods = {"development": (dates[0], dates[1]), "holdout": (dates[1], dates[2]), "combined": (dates[0], dates[2])}
    rows, candidate_sources = [], {}
    # Guard regeneration's shared feature functions against accidental network fallback.
    with patch("socket.socket.connect", side_effect=RuntimeError("Offline validation forbids network access")):
        for profile in profiles:
            if args.regenerate:
                candidates = regenerate_candidates(profile, histories, args.train_min, args.retrain_every)
                candidates.to_csv(args.out_dir / f"{profile.name}_regenerated_candidates.csv", index=False)
            else:
                candidates, candidate_sources[profile.name] = load_candidates(profile, args.trades_dir)
            for costs in COST_SCENARIOS:
                for period, (start, end) in periods.items():
                    eligible = candidates[(candidates.entry_time >= start) & (candidates.exit_time < end)].copy()
                    selected, stats = allocate_candidates(eligible, profile, costs, args.starting_equity)
                    curve, risk = mark_to_market(selected, histories, costs, profile.leverage, args.starting_equity)
                    stem = f"{profile.name}_{period}_{costs.name}"
                    selected.to_csv(args.out_dir / f"{stem}_selected_trades.csv", index=False)
                    curve.to_csv(args.out_dir / f"{stem}_equity.csv")
                    rows.append({"profile": profile.name, "period": period, "cost_scenario": costs.name, "period_start": str(start), "period_end_exclusive": str(end), **stats, **risk})
    summary = pd.DataFrame(rows)
    comparisons = {}
    default_holdout = args.holdout_start == "2024-01-01" and args.holdout_end == "2027-01-01"
    for name, expected in EXPECTED_HOLDOUT.items():
        row = summary[(summary.profile == name) & (summary.period == "holdout") & (summary.cost_scenario == "kraken_10k_high_margin")].iloc[0]
        matches = default_holdout and all(abs(float(row[key]) - value) <= (0 if key == "trades" else 0.11) for key, value in expected.items())
        comparisons[name] = "matches published rounded return, realized drawdown, and trade count" if matches else "does not match the published rounded holdout figures (inspect summary and input dates)"
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "regenerated standalone candidates with allocation replay" if args.regenerate else "saved standalone candidate allocation replay; model signals not independently regenerated",
        "starting_equity": args.starting_equity,
        "ohlc_sources": ohlc_sources, "candidate_sources": candidate_sources, "fear_greed_source": fng_source,
        "regeneration_settings": {"train_min": args.train_min, "retrain_every": args.retrain_every} if args.regenerate else None,
        "profiles": {profile.name: {"symbols": {symbol: asdict(strategy) for symbol, strategy in profile.symbols.items()}, "leverage": profile.leverage, "position_fraction": profile.position_fraction, "max_open": profile.max_open, "use_dynamic_threshold": profile.use_dynamic_threshold} for profile in profiles},
        "cost_scenarios": [asdict(costs) for costs in COST_SCENARIOS],
        "published_holdout_comparison": comparisons,
        "limitations": [
            "Saved candidates are not independent regeneration of model signals; their original training inputs and Fear & Greed snapshot are not preserved in the trade CSVs. Source hashes document the files used now, not their historical provenance.",
            "Standalone candidates skip same-coin signals while their own hypothetical trade is open, even when the portfolio rejects that trade. This is an allocation approximation, not a joint multi-asset signal simulation.",
            "Sizing uses realized equity (33% per position, 2x leverage, maximum three positions) to reproduce the report. Intratrade equity changes do not alter those allocation decisions or trigger margin calls/liquidations.",
            "Hourly-close drawdown includes accrued margin charges and estimated closing/execution costs. The adverse-hour proxy sums each coin's hourly low; lows may occur at different instants. Tick-level loss paths and liquidation thresholds are not simulated.",
            "Some local OHLC series have missing hours. Missing open-position marks carry the last observed close (see missing_open_position_mark_hours in summary), so risk can be understated. Original realized fees count bars rather than elapsed time; extra_elapsed_rollover_cost_cash shows the additional elapsed-time charge on accepted positions without recomputing compounded allocations.",
            "Entry occurs at the signal candle close; maker fills, queue position, latency, minimum order sizes, exchange pair eligibility, euro/USD/USDT currency conversion, and taxes are not modeled. A small real account may not execute every accepted candidate.",
            "Fee scenarios reproduce explicit historical report assumptions. They are not a verification of current exchange fees or account tier. The zero-volume scenario is more relevant than assuming a $10k tier for a small account.",
            "Date intervals include only candidates whose entry and exit both fall inside the interval. Cross-boundary positions are excluded. The 2026 period ends at each local cache's actual last bar, not at the requested 2027 bound.",
            "The 2024 onward data was already inspected during the earlier strategy recommendation; it is no longer a new untouched holdout. Comparing or selecting strategies on these results adds selection bias. Fresh paper observations are needed.",
            "Saved candidates for both portfolios used a dynamic floor that can raise the stated base probability thresholds. The six-coin paper profile retains it; the three-coin paper profile uses fixed thresholds to match the current live bot. Optional regeneration follows each current profile; the saved baseline comparison therefore does not exactly reproduce the current live strategy.",
        ],
    }
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    write_report(args.out_dir, summary, manifest)
    print(summary[["profile", "period", "cost_scenario", "trades", "return_pct", "realized_drawdown_pct", "close_mtm_drawdown_pct", "adverse_hour_drawdown_pct"]].to_string(index=False))
    print(f"\nReport: {args.out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
