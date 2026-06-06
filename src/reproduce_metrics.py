"""Reproduce the public repo's headline metrics from packaged audit artifacts.

This script intentionally reads only CSV files committed in this repository.
It does not load raw vendor market data, private models, API keys, or broker
connections. For a full raw-data audit rerun, see overfit_replication_audit.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "overfit_audit"
TABLES = AUDIT / "tables"


def row(df: pd.DataFrame, variant: str, slippage: float | None = None) -> pd.Series:
    rows = df[df["variant"] == variant]
    if slippage is not None:
        rows = rows[rows["slippage_points_per_leg"].astype(float) == slippage]
    if rows.empty:
        raise SystemExit(f"Missing audit row: {variant}, slippage={slippage}")
    return rows.iloc[0]


def money(value: object) -> str:
    return f"${float(value):+,.2f}"


def main() -> None:
    metrics = pd.read_csv(TABLES / "audit_metrics.csv")
    claims = pd.read_csv(AUDIT / "claim_matrix.csv")
    bootstrap = pd.read_csv(TABLES / "bootstrap_summary.csv")
    leakage = pd.read_csv(TABLES / "leakage_audit.csv")
    stress = pd.read_csv(TABLES / "stress_tests.csv")

    replicated = row(metrics, "independent_straddle_slippage_stress", 1.5)
    clean = row(metrics, "clean_eval_straddle_slippage_stress", 1.5)
    purged = row(metrics, "purged_clean_eval_straddle_slippage_stress", 1.5)
    label_shuffle = row(metrics, "label_shuffle_straddle_slippage_stress", 1.5)
    severe_slippage = row(metrics, "independent_straddle_slippage_stress", 2.5)
    adverse = row(stress, "adverse_same_bar_ordering", 1.5)

    print("NQ Futures ML Risk Audit - packaged metric reproduction")
    print("=" * 64)
    print(f"Replicated original 1.5 pt/leg: {int(replicated['trades']):,} trades, {money(replicated['total_pnl'])}, monthly Sharpe {float(replicated['monthly_sharpe_ann']):.3f}")
    print(f"Clean-eval 1.5 pt/leg:          {int(clean['trades']):,} trades, {money(clean['total_pnl'])}, monthly Sharpe {float(clean['monthly_sharpe_ann']):.3f}")
    print(f"Purged clean-eval 1.5 pt/leg:   {int(purged['trades']):,} trades, {money(purged['total_pnl'])}, monthly Sharpe {float(purged['monthly_sharpe_ann']):.3f}")
    print(f"Label-shuffle null 1.5 pt/leg:  {int(label_shuffle['trades']):,} trades, {money(label_shuffle['total_pnl'])}")
    print(f"2.5 pt/leg failure boundary:    {money(severe_slippage['total_pnl'])}")
    print(f"Same-bar adverse ordering:      {money(adverse['total_pnl'])}")
    print()
    print("Claim matrix verdicts")
    print(claims["verdict"].value_counts().to_string())
    print()
    print("Leakage audit")
    print(leakage[["check", "verdict"]].to_string(index=False))
    print()
    print("Block bootstrap summary")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
