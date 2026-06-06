from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "overfit_audit"
TABLES = AUDIT / "tables"


def metric_row(df: pd.DataFrame, variant: str, slippage: float | None = None) -> pd.Series:
    rows = df[df["variant"] == variant]
    if slippage is not None:
        rows = rows[rows["slippage_points_per_leg"].astype(float) == slippage]
    assert not rows.empty, f"missing row for {variant} {slippage}"
    return rows.iloc[0]


def test_conservative_result_is_replicated() -> None:
    metrics = pd.read_csv(TABLES / "audit_metrics.csv")
    row = metric_row(metrics, "independent_straddle_slippage_stress", 1.5)

    assert int(row["trades"]) == 2022
    assert float(row["total_pnl"]) == 61301.12
    assert float(row["monthly_sharpe_ann"]) == 2.848


def test_clean_and_purged_validation_remain_positive() -> None:
    metrics = pd.read_csv(TABLES / "audit_metrics.csv")
    clean = metric_row(metrics, "clean_eval_straddle_slippage_stress", 1.5)
    purged = metric_row(metrics, "purged_clean_eval_straddle_slippage_stress", 1.5)

    assert float(clean["total_pnl"]) > 0
    assert float(purged["total_pnl"]) > 0
    assert int(clean["trades"]) >= 1900
    assert int(purged["trades"]) >= 1900


def test_null_and_failure_cases_are_visible() -> None:
    metrics = pd.read_csv(TABLES / "audit_metrics.csv")
    stress = pd.read_csv(TABLES / "stress_tests.csv")
    cross = pd.read_csv(TABLES / "cross_instrument_sanity.csv")

    label_null = metric_row(metrics, "label_shuffle_straddle_slippage_stress", 1.5)
    severe_slippage = metric_row(metrics, "independent_straddle_slippage_stress", 2.5)
    adverse = metric_row(stress, "adverse_same_bar_ordering", 1.5)

    assert int(label_null["trades"]) == 0
    assert float(severe_slippage["total_pnl"]) < 0
    assert float(adverse["total_pnl"]) < 0
    assert (cross["total_pnl"].astype(float) < 0).all()


def test_claim_matrix_rejects_old_sharpe_language() -> None:
    claims = pd.read_csv(AUDIT / "claim_matrix.csv")
    old_sharpe = claims[claims["claim"].str.contains("finance Sharpe", case=False, na=False)]

    assert not old_sharpe.empty
    assert set(old_sharpe["verdict"]) == {"rejected"}


def test_leakage_audit_has_no_feature_future_shift_failure() -> None:
    leakage = pd.read_csv(TABLES / "leakage_audit.csv")
    feature_check = leakage[leakage["check"] == "future_shift_in_features"]

    assert not feature_check.empty
    assert feature_check.iloc[0]["verdict"] == "pass"

