# Methods

This project audits an NQ futures research strategy. The point is not to prove a live trading edge. The point is to show a disciplined process for testing whether a backtest claim survives stricter validation.

## Research Question

Can a machine-learning volatility gate improve a synthetic NQ straddle-style backtest after accounting for transaction costs, walk-forward validation, slippage stress, and failure cases?

## Data

The original audit used Databento-derived 1-minute futures bars with these fields:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

Raw vendor data is not committed. The committed repository contains the audit outputs, charts, reports, and summary tables needed to reproduce the packaged evidence.

## Strategy Logic

The strategy is not a pure directional forecast.

At each candidate bar:

1. Build historical-only technical and volatility features.
2. Train a LightGBM classifier to estimate whether the next bar is likely to have enough range expansion.
3. Trade only when the volatility gate, regular-session filter, ATR expansion filter, and confidence threshold pass.
4. Simulate a synthetic straddle-style position with one long and one short leg.
5. Exit each leg using ATR-scaled profit targets and stop losses.

The directional model is included as a comparison case. It failed in the audit, which is why the project is framed around volatility/risk gating rather than direction prediction.

## Walk-Forward Design

The original walk-forward process used:

- 12-month training window
- 3-month test window
- 3-month step size
- 5 test folds

The audit also added:

- clean-eval rerun, where early stopping uses a train-tail validation slice instead of the test fold
- purged clean-eval rerun, where train/test boundary overlap is removed

## Overfitting Tests

The audit checks overfitting from multiple angles:

| Test | Purpose | Result |
|---|---|---|
| Independent rebuild | Verify original claims without importing original strategy functions | Reproduced 1.5 pt/leg conservative result |
| Clean-eval rerun | Remove test-fold early stopping issue | Stayed positive |
| Purged clean-eval rerun | Remove fold-boundary overlap | Stayed positive |
| Label-shuffle null | Test whether the model still trades after target randomization | Produced zero trades |
| Feature ablation | Check whether the result only comes from one feature family | Volatility features carried most of the result |
| Slippage stress | Test execution-cost sensitivity | Failed at 2.5 pt/leg |
| Block bootstrap | Preserve clustered trade blocks in Monte Carlo | Stayed positive in packaged historical blocks |
| Cross-instrument check | Test broad futures generalization | Failed on ES and MGC |

## Main Results

At 1.5 NQ points of slippage per leg:

| Validation | Trades | PnL | Monthly Ann. Sharpe |
|---|---:|---:|---:|
| Replicated original logic | 2,022 | +$61,301 | 2.85 |
| Clean-eval rerun | 1,925 | +$60,511 | 2.91 |
| Purged clean-eval rerun | 1,923 | +$56,145 | 2.87 |

## Failure Cases

The failures are central to the project:

- The old 9+ or 13+ "Sharpe" value was rejected because it was a trade-level t-statistic.
- Directional prediction lost money.
- Same-bar adverse fill ordering produced a large loss.
- 2.5 pt/leg slippage made the strategy negative.
- Cross-instrument checks failed on ES and MGC.

## Correct Interpretation

This is a strong NQ-specific historical research and validation project. It is not a proven trading system.

The correct next step is a locked forward paper-trading period with no parameter changes.

