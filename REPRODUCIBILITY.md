# Reproducibility

This public repository includes the packaged audit outputs, charts, reports, and the source code used to produce them.

## Quick Check

Run this command to verify the included metrics from committed CSV artifacts:

```bash
python src/reproduce_metrics.py
```

This does not require raw vendor market data.

## Full Raw-Data Audit

The full audit runner is:

```bash
python src/overfit_replication_audit.py
```

The raw 1-minute futures data is not committed. To rerun from raw bars, provide the same OHLCV CSV layout used in the audit:

- timestamp
- open
- high
- low
- close
- volume

The original local audit used Databento-derived 1-minute futures bars. Raw vendor data, API keys, broker config, and model pickle files are intentionally excluded.

## What Is Reproducible Here

- Claim matrix verdicts.
- Packaged headline metrics.
- Leakage audit summary.
- Walk-forward fold summaries.
- Slippage stress results.
- Block-bootstrap summaries.
- Feature ablation summaries.
- Cross-instrument failure checks.

## What Is Not Claimed

This is not a live trading bot, a production strategy, or a guarantee of future profitability. The audit explicitly documents execution and validation failure modes.

