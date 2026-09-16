# Six-coin strategy validation and paper comparison

This compares the proposed `six_coin_v2` portfolio with the existing
`three_coin_v2` strategy using simulated money. It uses public market data,
requires no Kraken credentials, and sends no Telegram messages. Each paper
portfolio starts with **170 USD of simulated equity**, independent of the real
account balance; this is not a snapshot of the user's Kraken account. The
existing live strategy and scheduler remain unchanged. This implements the
agreed validation and paper comparison before using the candidate with real
money. There is no automatic promotion to live trading.

## Candidate configuration

| Market | Maximum hold (hours) | Entry probability floor |
| --- | ---: | ---: |
| ADA-USD | 24 | 0.70 |
| DOGE-USD | 48 | 0.68 |
| LINK-USD | 72 | 0.70 |
| SOL-USD | 72 | 0.65 |
| XLM-USD | 72 | 0.70 |
| XRP-USD | 72 | 0.65 |

The candidate enables the model's dynamic entry threshold, matching the saved
research: the values above are floors, and the effective threshold can be
higher. The `three_coin_v2` baseline uses ADA, DOGE, and SOL with 24-hour holds
and a fixed 0.70 entry threshold, matching the current live workflow.

Both portfolios are long-only, exit below probability 0.40 or at their holding
limit, target 33% of total account equity per position, and permit at most three
positions at 2x leverage. Minimum order size, available margin, and fees can
change the simulated position size. Both use daily Fear & Greed features and
the strategy's Fear & Greed filter.

## Reproduce the historical comparison

With the project dependencies installed, run:

```sh
python research_six_coin_validation.py --starting-equity 170 --out-dir validation_outputs/six_coin
```

Read `validation_outputs/six_coin/REPORT.md` alongside `summary.csv` and
`manifest.json`. The directory also contains selected trades and equity series
for each profile, period, and cost scenario. The default run replays saved
candidate trades from `research_outputs/nine_coin_optimizer`; it does not
independently regenerate the trained models. Consult `--help` for regeneration
options and required historical Fear & Greed input.

Pay attention to portfolio drawdown including open positions, costs, overlapping
trades, and sample size. The previously quoted historical return is a research
result rather than a forecast; replaying the same selected strategy does not
turn that period into an independent holdout.

## Run the paper comparison

```sh
python -m unittest discover -s tests -v
python ml_portfolio_paper.py --state-file paper_comparison_state.json --report-file paper_comparison_report.md --start-equity 170
```

The default run advances both named portfolios and writes a comparison report.
Keep the state file between runs so positions, costs, and performance carry
forward. The 170 USD starting amount initializes a new account; it does not
add funds on every run. Back up the state before intentionally starting a new
experiment with a different `paper_comparison*.json` state filename. Changing
the saved configuration requires a separate experiment.

Live paper scans need public Yahoo hourly candles, Kraken bid/ask quotes and
pair metadata, and Alternative.me Fear & Greed history. Missing network access
or data can prevent a scan. Signals use completed hourly candles; delayed scans
do not reconstruct missed entries or exits.

The simulated execution includes taker costs, spread, slippage, and margin
rollover assumptions. Taker fees are 0.80% per side, matching the published
[Tier 1 schedule](https://www.kraken.com/features/fee-schedule) checked on
2026-09-16. Opening margin and each elapsed four-hour rollover use 0.04% of
entry notional; slippage is 0.05% per side in addition to the observed spread.
The model's signal estimates retain the existing strategy cost inputs for
comparison, while simulated P&L charges the higher execution costs above.
These are explicit assumptions, not observations of actual
maker fills, exchange liquidity, or achievable profit. Assess the candidate
against the baseline over new observations before considering a live change.
Equity and drawdown include open positions at each observed scan, but miss
price moves between scans. Liquidations and account-specific margin eligibility
are not modeled.

The cached replay reproduced the previously reported six-coin cumulative
holdout return (+148.94%, 31 trades), but hourly marks exposed a **37.33%**
drawdown versus the report's 19.09% realized-only drawdown. Development-period
hourly drawdown was 72.48%. These numbers use the report's $10k-volume cost
scenario and exclude liquidation effects. They reinforce the need for paper
validation; they are not a live-deployment approval or a return forecast.

## GitHub Actions operation

`.github/workflows/ml-six-coin-paper.yml` runs at minutes 7, 22, 37, and 52 of
each hour and supports manual **Run workflow**. Scheduled workflows start once
the file is on the default branch and Actions is enabled. GitHub cron is best
effort: jobs can be delayed or skipped, so this is not a guaranteed 15-minute
execution service.

Each successful run saves only `paper_comparison_state.json` under the separate
`ml-six-coin-paper-state-v1-` cache prefix. It does not read or write live state
or live cache. Restore errors fail the job; an absent or evicted cache starts
a new experiment. Use the report's starting timestamp to detect resets and
restore an archived paper state when continuity is needed.

The run summary shows the report. Artifacts named
`six-coin-paper-<run_id>-<run_attempt>` archive the state and report for up to
90 days, subject to repository retention limits. Download important checkpoints
before they expire. No real orders are scheduled by this workflow.
