# Locked Paper-Trading Log

This folder is for forward validation after the historical audit.

Rules:

- No parameter changes during a logging window.
- Record every signal, including skipped trades.
- Record the expected simulated entry and the paper/broker fill.
- Record slippage and whether the fill matched the backtest assumption.
- Do not summarize performance without preserving the row-level log.

The goal is to test whether the historical assumptions survive live market conditions, not to keep optimizing the backtest.

## Files

- `paper_trading_log_template.csv` - row-level signal and fill log.
- `weekly_review_template.md` - weekly review format.

