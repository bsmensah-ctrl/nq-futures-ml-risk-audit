# Validation Checklist

This checklist maps the core claims to the evidence in the repository.

| Claim | Evidence | Status |
|---|---|---|
| The old high Sharpe claim was mislabeled | `overfit_audit/claim_matrix.csv` | Rejected as finance Sharpe |
| Independent rebuild matched the conservative result | `overfit_audit/tables/audit_metrics.csv` | Supported |
| Clean validation stayed positive | `clean_eval_straddle_slippage_stress` rows in `audit_metrics.csv` | Supported |
| Purged validation stayed positive | `purged_clean_eval_straddle_slippage_stress` rows in `audit_metrics.csv` | Supported |
| Direct future-looking feature leakage was not found | `overfit_audit/tables/leakage_audit.csv` | Passed static check |
| Test-fold early stopping was a validation weakness | `leakage_audit.csv` | Corrected with clean-eval rerun |
| Label-shuffle null did not preserve the result | `label_shuffle_straddle_slippage_stress` rows | Supported |
| Execution cost can break the edge | 2.5 pt/leg rows in `audit_metrics.csv` | Supported |
| Intrabar fill ordering is a major unresolved risk | `adverse_same_bar_ordering` in `stress_tests.csv` | Supported |
| Broad futures generalization is not supported | `cross_instrument_sanity.csv` | Failed ES and MGC |

## What Would Upgrade The Evidence

- Locked forward paper-trading log.
- Broker or NinjaTrader replay fill comparison.
- Separate raw-data reproduction bundle using redistributable sample data.
- More instruments and market regimes.
- Smaller modules around feature building, walk-forward splitting, simulation, and reporting.

## What This Repo Should Not Claim

- Live profitability.
- Production readiness.
- A broad futures edge.
- A finance Sharpe ratio based on trade-level t-statistics.

