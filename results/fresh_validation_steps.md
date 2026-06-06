# Fresh Conservative Validation Steps

- Run timestamp UTC: 2026-06-05T23:03:41.390905+00:00
- Source script: `original_private_research/GBM_Straddle_Databento.py`
- Source script SHA256: `d6def243943a86fa2d275484df5299f77b716c9778c661f395bc308605f5c8d4`
- Source data: `data/raw/databento_nq_1m.csv`
- Source data SHA256: `860e14b06b44a3d1b8064b3aeb0306e47164b3f48fe7a73b85b2736b367f6275`
- Python: `python`
- Platform: `Windows-10-10.0.26200-SP0`

## What Was Rerun

1. Loaded raw Databento-derived NQ 1-minute bars from the original project.
2. Rebuilt the same feature set used by `GBM_Straddle_Databento.py`.
3. Reran 12-month train / 3-month test / 3-month slide walk-forward validation.
4. Saved timestamped trade streams for filtered straddle, delayed-entry straddle, and directional comparison.
5. Recomputed monthly annualized Sharpe from monthly PnL instead of using the old trade-level t-statistic.
6. Applied slippage stress from 0.5 to 2.5 NQ points per leg.

## Key Correction

The old `sh` field is a trade-level t-statistic: `(mean_trade_pnl / std_trade_pnl) * sqrt(n_trades)`. It is useful internally, but it is not a finance-style Sharpe ratio. Portfolio-facing materials should cite monthly annualized Sharpe and execution-stressed results.

