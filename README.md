# Kraken Trading Bot

This repo runs a small Kraken trading setup from GitHub Actions.

Important: this can place real margin orders on Kraken. Use dry-run when testing.

## What Is Live

### ML V2 Live Trader

File: `ml_live_trade.py`

Workflow: `.github/workflows/ml-live-trade.yml`

Runs every 15 minutes in live mode with real money. This is the only scheduled
trading controller. For a true always-on VPS/local process, run
`ml_live_trade.py` with `ML_LIVE_RUN_FOREVER=true`.

**Profit-tuned profile (current live default):** scans frequently, but avoids
forcing weak trades. A sweep of the fixed-threshold live mode found that lower
thresholds traded more but lost money after fees; the current settings favored
higher after-fee expectancy. This is not a guarantee of daily profit.

Coins: `SOL-USD, LINK-USD, DOGE-USD, XRP-USD, ADA-USD`.

How it trades:

- Long-only (shorts historically destroyed equity in regime research).
- Buy when `prob_up > 0.65` with a **fixed** threshold (dynamic breakeven raise
  disabled via `ML_LIVE_USE_DYNAMIC_THRESHOLD=false`).
- Fear & Greed is used as a model feature, but the hard 25–40 fear bucket
  filter is off.
- Each trade uses a flat 20% of usable margin at 2x leverage (no confidence
  sizing tiers).
- On small accounts, size is bumped up to Kraken's minimum order when affordable.
- Hold ~2 days (48 x 1h bars), then close on a time-based exit; early exit if
  model `prob_up` drops below 0.40.
- At most one position per coin, up to 5 open positions.
- Entries use market orders so unfilled maker limits do not skip trades.
- Exits are always market orders (a time-boxed strategy must be able to get out).
- Before opening new live positions, the bot checks Kraken's open positions so
  a missing or stale `ml_live_state.json` does not create duplicate exposure in
  the same coin.

State (which positions we opened and when to close them) is saved to
`ml_live_state.json` and cached between runs so it survives independent cron runs.

The lower-threshold high-frequency profile (thr 0.55, 24h holds) remains easy
to reproduce with environment overrides, but the recent sweep showed it had
negative after-fee expectancy. It is not the live default.

To test safely, run the workflow manually with `dry_run=true` — it runs the full
logic without placing real orders.

### Daily Telegram Portfolio Report

File: `daily_portfolio_report.py`

Workflow: `.github/workflows/daily-portfolio-report.yml`

Runs every day at 09:00 Asia/Seoul and sends a Telegram portfolio update.

### Manual Workflows

These are not scheduled automatically:

- `.github/workflows/trading-bot-v4.yml` (old V4 swing bot)
- paper-trading workflows

Use them manually only when needed.

## Required GitHub Secrets

Set these in GitHub:

`Settings -> Secrets and variables -> Actions`

Required:

```text
KRAKEN_API_KEY
KRAKEN_API_SECRET
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

Optional:

```text
NEWSDATA_API_KEY
CRYPTOCOMPARE_API_KEY
```

Do not commit real keys to the repo.

## Common Commands

Run the ML V2 live trader manually in live mode (real money):

```bash
gh workflow run ml-live-trade.yml \
  --repo JKang78/Trading-Bot-V4 \
  --ref main \
  -f dry_run=false \
  -f strategy=v2
```

Run the ML V2 live trader safely in GitHub dry-run mode (no real orders):

```bash
gh workflow run ml-live-trade.yml \
  --repo JKang78/Trading-Bot-V4 \
  --ref main \
  -f dry_run=true \
  -f strategy=v2
```

Run the daily portfolio report manually:

```bash
gh workflow run daily-portfolio-report.yml \
  --repo JKang78/Trading-Bot-V4 \
  --ref main
```

Run locally in dry-run mode:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
ML_LIVE_DRY_RUN=true venv/bin/python ml_live_trade.py
```

Run continuously on a local machine or VPS (real money if `ML_LIVE_DRY_RUN=false`):

```bash
ML_LIVE_RUN_FOREVER=true \
ML_LIVE_LOOP_INTERVAL_SEC=900 \
ML_LIVE_DRY_RUN=false \
venv/bin/python ml_live_trade.py
```

Run for exactly 24 hours, then exit:

```bash
ML_LIVE_RUN_FOREVER=true \
ML_LIVE_MAX_RUNTIME_HOURS=24 \
ML_LIVE_DRY_RUN=false \
venv/bin/python ml_live_trade.py
```

Run the standard ML V2 validation backtest that matches the live profile:

```bash
venv/bin/python ml_strategy_backtest.py \
  --symbols SOL-USD,LINK-USD,DOGE-USD,XRP-USD,ADA-USD \
  --period 720d \
  --horizon 48 \
  --buy-thr 0.65 \
  --sell-thr 0 \
  --exit-thr 0.40 \
  --fee 0.0040 \
  --entry-fee 0.0040 \
  --exit-fee 0.0040 \
  --margin-open-fee 0.0002 \
  --rollover-fee 0.0002 \
  --no-dynamic-threshold \
  --out ml_strategy_trades_live_v2.csv
```

Run the long-history profit harness:

```bash
venv/bin/python research_long_history_profit_harness.py --reuse-trades
```

Run the V2/V3 fee-sensitivity research sweep:

```bash
venv/bin/python research_v2_v3_profitability.py
```

Research the old V4 bear-market short profile with margin fees included:

```bash
venv/bin/python backtest.py \
  --symbols BTC-USD,ETH-USD \
  --period 720d \
  --interval 1h \
  --leverage 2 \
  --directions short \
  --trend-ema 200 \
  --fee 0.0040 \
  --exit-fee 0.0080 \
  --margin-open-fee 0.0004 \
  --rollover-fee-4h 0.0004 \
  --spread-slippage-buffer 0.0015 \
  --min-confidence 0.80 \
  --max-signal-age-hours 12 \
  --min-expectancy-pct 0.25 \
  --out v4_short_bear_research.csv
```

## Main Files

```text
ml_live_trade.py                ML V2 live trader (scheduled, real money)
ml_strategy.py                  ML V2/V3 strategy logic
kraken_bot_v4_advanced.py       old V4 swing bot (manual only)
research_v2_v3_profitability.py V2/V3 walk-forward fee-sensitivity research
daily_portfolio_report.py       Telegram portfolio report
.github/workflows/              GitHub schedules and manual workflows
.env.example                    local environment template
```

## State Files

These files are generated and should not be committed:

```text
ml_live_state.json
v4_position_state.json
rl_state.json
.env
```

## Current Operating Model

GitHub Actions is the scheduled runtime.

The ML V2 live trader (`ml-live-trade.yml`) is the only scheduled trading
controller and runs every 15 minutes as live by default. The old V4 bot and
paper-trading workflows can still be run manually, but their automatic schedules
are paused to avoid two bots trading the same Kraken account at the same time.
