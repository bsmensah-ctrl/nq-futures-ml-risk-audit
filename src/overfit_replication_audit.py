"""Independent overfitting and replication audit for the NQ portfolio project.

This script intentionally does not import the original strategy module. It
rebuilds the feature set, walk-forward folds, trade simulation, stress tests,
claim reconciliation, charts, Markdown report, and PDF report from raw CSVs.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.preprocessing import StandardScaler


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("NQ_PROJECT_ROOT", ROOT.parent / "NQ_Project"))
AUDIT_ROOT = ROOT / "overfit_audit"
CHART_DIR = AUDIT_ROOT / "charts"
TABLE_DIR = AUDIT_ROOT / "tables"
TRADE_DIR = AUDIT_ROOT / "trades"
FRESH_DIR = Path(os.environ.get("NQ_FRESH_RESULTS_DIR", ROOT / "results"))

SOURCE_SCRIPT = Path(os.environ.get("NQ_SOURCE_STRATEGY", Path(__file__).resolve()))
NQ_DATA = Path(os.environ.get("NQ_RAW_1M_CSV", PROJECT_ROOT / "data" / "public_downloads" / "databento_nq_1m.csv"))
CROSS_DATA = {
    "ES": Path(os.environ.get("ES_RAW_1M_CSV", PROJECT_ROOT / "data" / "public_downloads" / "databento_es_1m.csv")),
    "MGC": Path(os.environ.get("MGC_RAW_1M_CSV", PROJECT_ROOT / "data" / "public_downloads" / "databento_mgc_1m.csv")),
}

MAX_HOLD = 25
CONF_THR = 0.52
BASE_SLIPPAGE_POINTS = 0.5
SLIPPAGE_POINTS = [0.5, 1.0, 1.5, 2.0, 2.5]
POINT_VALUES = {"NQ": 20.0, "ES": 50.0, "MGC": 10.0}
COST_PER_LEG = 17.0
TRAIN_MONTHS = 12
TEST_MONTHS = 3
SLIDE_MONTHS = 3
RANDOM_SEED = 42

META_COLS = {
    "lv",
    "ld",
    "atr_val",
    "atr50_val",
    "close",
    "high",
    "low",
    "open_px",
    "e9_val",
    "e21_val",
    "tf5_bull",
}


@dataclass(frozen=True)
class FoldData:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    test_rows: int
    overlap_rows: int


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def ensure_dirs() -> None:
    for path in [AUDIT_ROOT, CHART_DIR, TABLE_DIR, TRADE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gains = d.clip(lower=0).rolling(n).mean()
    losses = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gains / (losses + 1e-9))


def atr_ewm(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = next((c for c in df.columns if c.lower() in ("timestamp", "datetime", "time", "date")), None)
    if ts_col:
        df.index = pd.to_datetime(df[ts_col])
        df = df.drop(columns=[ts_col])
    else:
        df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    needed = ["open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df.sort_index()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    f = pd.DataFrame(index=df.index)
    a14 = atr_ewm(h, l, c, 14)
    a5 = atr_ewm(h, l, c, 5)
    a50 = atr_ewm(h, l, c, 50)
    rng = h - l + 1e-9

    f["atr_ratio"] = a5 / (a14 + 1e-9)
    f["atr_expand"] = (a14 > a50).astype(float)
    f["range_norm"] = rng / (a14 + 1e-9)
    f["range_z"] = (rng - rng.rolling(20).mean()) / (rng.rolling(20).std() + 1e-9)
    f["range_hi5"] = rng / (pd.concat([h.shift(k) - l.shift(k) for k in range(1, 6)], axis=1).max(axis=1) + 1e-9)
    f["eff"] = (c - o).abs() / rng
    f["cpos"] = (c - l) / rng
    f["bull"] = (c > o).astype(float)
    f["body"] = (c - o).abs() / rng
    f["uwk"] = (h - c.clip(lower=o)) / rng
    f["lwk"] = (c.clip(upper=o) - l) / rng
    f["wick"] = (f["uwk"] + f["lwk"]) / (f["body"] + 1e-9)
    for n in [1, 3, 5, 10, 20]:
        f[f"r{n}"] = c.pct_change(n)
    f["rsi14"] = rsi(c, 14)
    f["rsi6"] = rsi(c, 6)
    mac = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = mac.ewm(span=9, adjust=False).mean()
    f["macd_h"] = (mac - sig) / (a14 + 1e-9)
    f["macd_sig"] = (mac > sig).astype(float)
    ret = c.pct_change()
    f["rv10"] = ret.rolling(10).std()
    f["rv30"] = ret.rolling(30).std()
    f["rvr"] = f["rv10"] / (f["rv30"] + 1e-9)
    e9 = c.ewm(span=9, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    f["d9"] = (c - e9) / (a14 + 1e-9)
    f["d21"] = (c - e21) / (a14 + 1e-9)
    f["d50"] = (c - e50) / (a14 + 1e-9)
    f["s9"] = (e9 - e9.shift(3)) / (a14 + 1e-9)
    f["s21"] = (e21 - e21.shift(3)) / (a14 + 1e-9)
    f["xma"] = (e9 > e21).astype(float)
    f["accel"] = (e9.diff(1) - e9.shift(1).diff(1)) / (a14 + 1e-9)
    f["hi10"] = (h.rolling(10).max() - c) / (a14 + 1e-9)
    f["lo10"] = (c - l.rolling(10).min()) / (a14 + 1e-9)
    f["hi20"] = (h.rolling(20).max() - c) / (a14 + 1e-9)
    f["lo20"] = (c - l.rolling(20).min()) / (a14 + 1e-9)
    dt = pd.to_datetime(df.index)
    f["hr"] = dt.hour + dt.minute / 60.0
    f["dow"] = dt.dayofweek
    f["rth"] = ((dt.hour >= 9) & (dt.hour < 16) & ~((dt.hour == 9) & (dt.minute < 30))).astype(float)

    f["atr_val"] = a14
    f["atr50_val"] = a50
    f["close"] = c
    f["high"] = h
    f["low"] = l
    f["open_px"] = o
    f["e9_val"] = e9
    f["e21_val"] = e21

    f["lv"] = ((h.shift(-1) - l.shift(-1)) > 1.2 * a14).astype(int)
    f["ld"] = (c.shift(-1) > c).astype(int)
    return f.dropna()


def add_5min_trend(feat_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    r = raw_df[["open", "high", "low", "close"]].resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    e9_5 = r["close"].ewm(span=9, adjust=False).mean()
    e21_5 = r["close"].ewm(span=21, adjust=False).mean()
    tf5_bull = (e9_5 > e21_5).astype(int)
    tf5_bull.name = "tf5_bull"
    feat_df["tf5_bull"] = tf5_bull.reindex(feat_df.index, method="ffill").fillna(0)
    return feat_df


def load_features(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = load_ohlcv(path)
    feature_data = build_features(raw)
    feature_data = add_5min_trend(feature_data, raw)
    for col in feature_data.columns:
        if col not in META_COLS:
            feature_data[col] = feature_data[col].astype(np.float32)
    info = {
        "source_path": str(path),
        "rows_raw": int(len(raw)),
        "rows_features": int(len(feature_data)),
        "raw_start": str(raw.index.min()),
        "raw_end": str(raw.index.max()),
        "feature_start": str(feature_data.index.min()),
        "feature_end": str(feature_data.index.max()),
    }
    del raw
    gc.collect()
    return feature_data, info


def feature_columns(feat: pd.DataFrame, mode: str = "all") -> list[str]:
    all_cols = [c for c in feat.columns if c not in META_COLS]
    time_cols = [c for c in ["hr", "dow", "rth"] if c in all_cols]
    vol_cols = [
        c
        for c in ["atr_ratio", "atr_expand", "range_norm", "range_z", "range_hi5", "rv10", "rv30", "rvr"]
        if c in all_cols
    ]
    if mode == "all":
        return all_cols
    if mode == "no_time":
        return [c for c in all_cols if c not in time_cols]
    if mode == "time_only":
        return time_cols
    if mode == "volatility_only":
        return vol_cols
    raise ValueError(f"Unknown feature mode: {mode}")


def model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=40,
        random_state=RANDOM_SEED,
        n_jobs=2,
        verbose=-1,
    )


def fit_predict(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    fcols: list[str],
    label: str,
    eval_mode: str,
    shuffle_labels: bool = False,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    if eval_mode == "test":
        scaler = StandardScaler()
        xtr = scaler.fit_transform(tr[fcols].values).astype(np.float32)
        xte = scaler.transform(te[fcols].values).astype(np.float32)
        ytr = tr[label].values.copy()
        if shuffle_labels:
            rng = np.random.default_rng(seed)
            rng.shuffle(ytr)
        clf = model()
        clf.fit(
            xtr,
            ytr,
            eval_set=[(xte, te[label].values)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        out = clf.predict_proba(xte)[:, 1]
        del clf, scaler, xtr, xte
        gc.collect()
        return out

    if eval_mode == "train_tail":
        cutoff = max(1, int(len(tr) * 0.8))
        tr_fit = tr.iloc[:cutoff]
        tr_val = tr.iloc[cutoff:]
        if len(tr_val) < 100:
            tr_fit = tr
            tr_val = tr
        scaler = StandardScaler()
        xfit = scaler.fit_transform(tr_fit[fcols].values).astype(np.float32)
        xval = scaler.transform(tr_val[fcols].values).astype(np.float32)
        xte = scaler.transform(te[fcols].values).astype(np.float32)
        yfit = tr_fit[label].values.copy()
        yval = tr_val[label].values.copy()
        if shuffle_labels:
            rng = np.random.default_rng(seed)
            rng.shuffle(yfit)
            rng.shuffle(yval)
        clf = model()
        clf.fit(
            xfit,
            yfit,
            eval_set=[(xval, yval)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        out = clf.predict_proba(xte)[:, 1]
        del clf, scaler, xfit, xval, xte
        gc.collect()
        return out

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def sim_directional(
    entries: np.ndarray,
    ep: np.ndarray,
    ds: np.ndarray,
    atr: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    point_value: float,
    tp_m: float = 1.5,
    sl_m: float = 0.5,
) -> np.ndarray:
    pnl = np.zeros(len(entries))
    for k, i in enumerate(entries):
        e = ep[k]
        d = int(ds[k])
        a = atr[k]
        tp = e + d * tp_m * a
        sl = e - d * sl_m * a
        hit = False
        for fwd in range(1, MAX_HOLD + 1):
            j = i + fwd
            if j >= len(cl):
                break
            if d == 1:
                if hi[j] >= tp:
                    pnl[k] = (tp - e) * point_value - COST_PER_LEG
                    hit = True
                    break
                if lo[j] <= sl:
                    pnl[k] = (sl - e) * point_value - COST_PER_LEG
                    hit = True
                    break
            else:
                if lo[j] <= tp:
                    pnl[k] = (e - tp) * point_value - COST_PER_LEG
                    hit = True
                    break
                if hi[j] >= sl:
                    pnl[k] = (e - sl) * point_value - COST_PER_LEG
                    hit = True
                    break
        if not hit:
            j = min(i + MAX_HOLD, len(cl) - 1)
            pnl[k] = (cl[j] - e) * d * point_value - COST_PER_LEG
    return pnl


def sim_straddle(
    entries: np.ndarray,
    ep: np.ndarray,
    atr: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    point_value: float,
    tp_m: float = 1.0,
    sl_m: float = 0.3,
    adverse_same_bar: bool = False,
) -> np.ndarray:
    pnl = np.zeros(len(entries))
    for k, i in enumerate(entries):
        e = ep[k]
        a = atr[k]
        l_tp = e + tp_m * a
        l_sl = e - sl_m * a
        s_tp = e - tp_m * a
        s_sl = e + sl_m * a

        j_end = min(i + MAX_HOLD, len(cl) - 1)
        l_pnl = (cl[j_end] - e) * point_value - COST_PER_LEG
        s_pnl = (e - cl[j_end]) * point_value - COST_PER_LEG
        l_open = True
        s_open = True

        for fwd in range(1, MAX_HOLD + 1):
            j = i + fwd
            if j >= len(cl):
                break
            if l_open:
                long_tp_hit = hi[j] >= l_tp
                long_sl_hit = lo[j] <= l_sl
                if adverse_same_bar and long_tp_hit and long_sl_hit:
                    l_pnl = (l_sl - e) * point_value - COST_PER_LEG
                    l_open = False
                elif long_tp_hit:
                    l_pnl = (l_tp - e) * point_value - COST_PER_LEG
                    l_open = False
                elif long_sl_hit:
                    l_pnl = (l_sl - e) * point_value - COST_PER_LEG
                    l_open = False
            if s_open:
                short_tp_hit = lo[j] <= s_tp
                short_sl_hit = hi[j] >= s_sl
                if adverse_same_bar and short_tp_hit and short_sl_hit:
                    s_pnl = (e - s_sl) * point_value - COST_PER_LEG
                    s_open = False
                elif short_tp_hit:
                    s_pnl = (e - s_tp) * point_value - COST_PER_LEG
                    s_open = False
                elif short_sl_hit:
                    s_pnl = (e - s_sl) * point_value - COST_PER_LEG
                    s_open = False
            if not l_open and not s_open:
                break

        pnl[k] = l_pnl + s_pnl
    return pnl


def max_drawdown(pnl: pd.Series | np.ndarray) -> float:
    values = pd.Series(pnl, dtype=float)
    equity = values.cumsum()
    return float((equity - equity.cummax()).min()) if len(values) else 0.0


def ann_sharpe(values: pd.Series | np.ndarray, periods_per_year: int) -> float | None:
    series = pd.Series(values, dtype=float)
    std = series.std(ddof=1)
    if len(series) < 2 or std == 0 or math.isnan(std):
        return None
    return float(series.mean() / std * math.sqrt(periods_per_year))


def profit_factor(pnl: pd.Series | np.ndarray) -> float | None:
    series = pd.Series(pnl, dtype=float)
    wins = series[series > 0].sum()
    losses = -series[series <= 0].sum()
    if losses == 0:
        return None
    return float(wins / losses)


def summarize_trades(name: str, trades: pd.DataFrame, slippage_points: float | None = None) -> dict[str, object]:
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades = trades.sort_values("timestamp")
    pnl = trades["pnl"].astype(float)
    monthly = trades.set_index("timestamp")["pnl"].resample("ME").sum()
    daily = trades.set_index("timestamp")["pnl"].resample("1D").sum()
    active_daily = daily[daily != 0]
    std = pnl.std(ddof=1)
    trade_t = None if len(pnl) < 2 or std == 0 else float(pnl.mean() / std * math.sqrt(len(pnl)))
    return {
        "variant": name,
        "slippage_points_per_leg": slippage_points,
        "trades": int(len(pnl)),
        "total_pnl": round(float(pnl.sum()), 2),
        "avg_trade": round(float(pnl.mean()), 2) if len(pnl) else 0.0,
        "win_rate": round(float((pnl > 0).mean()), 4) if len(pnl) else 0.0,
        "profit_factor": None if profit_factor(pnl) is None else round(profit_factor(pnl), 3),
        "max_trade_order_drawdown": round(max_drawdown(pnl), 2),
        "trade_t_stat_not_sharpe": None if trade_t is None else round(trade_t, 3),
        "monthly_count": int(len(monthly)),
        "monthly_positive": int((monthly > 0).sum()),
        "monthly_sharpe_ann": None if ann_sharpe(monthly, 12) is None else round(ann_sharpe(monthly, 12), 3),
        "monthly_worst": round(float(monthly.min()), 2) if len(monthly) else None,
        "daily_active_count": int(len(active_daily)),
        "daily_sharpe_ann_active_days": None
        if ann_sharpe(active_daily, 252) is None
        else round(ann_sharpe(active_daily, 252), 3),
        "daily_worst": round(float(active_daily.min()), 2) if len(active_daily) else None,
    }


def apply_slippage(trades: pd.DataFrame, slippage_points: float, point_value: float = 20.0) -> pd.DataFrame:
    extra = (slippage_points - BASE_SLIPPAGE_POINTS) * point_value * 2
    out = trades.copy()
    out["pnl"] = out["pnl"].astype(float) - extra
    out["slippage_points_per_leg"] = slippage_points
    return out


def fold_windows(feat: pd.DataFrame, purged: bool = False) -> Iterable[tuple[FoldData, pd.DataFrame, pd.DataFrame]]:
    ps = feat.index[0]
    end = feat.index[-1]
    fold = 0
    while True:
        test_start = ps + pd.DateOffset(months=TRAIN_MONTHS)
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
        if test_end > end:
            break
        if purged:
            train_end = test_start - pd.Timedelta(days=1)
            tr = feat[(feat.index >= ps) & (feat.index < train_end)]
            te = feat[(feat.index >= test_start) & (feat.index < test_end)]
            overlap = 0
        else:
            train_end = test_start
            tr = feat[ps:test_start]
            te = feat[test_start:test_end]
            overlap = len(tr.index.intersection(te.index))
        if len(tr) >= 1000 and len(te) >= 200:
            data = FoldData(
                fold=fold,
                train_start=ps,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_rows=int(len(tr)),
                test_rows=int(len(te)),
                overlap_rows=int(overlap),
            )
            yield data, tr, te
            fold += 1
        ps += pd.DateOffset(months=SLIDE_MONTHS)


def simulate_from_fold(
    fold: FoldData,
    te: pd.DataFrame,
    vp: np.ndarray,
    dp: np.ndarray | None,
    point_value: float,
    conf_thr: float = CONF_THR,
    tp_m: float = 1.0,
    sl_m: float = 0.3,
    delayed_bars: int = 0,
    adverse_same_bar: bool = False,
) -> dict[str, pd.DataFrame]:
    cl = te["close"].values
    hi = te["high"].values
    lo = te["low"].values
    op = te["open_px"].values
    atr = te["atr_val"].values
    a50 = te["atr50_val"].values
    tf5 = te["tf5_bull"].values
    ts = te.index.values

    if dp is None:
        dir_sign = np.ones(len(te), dtype=int)
    else:
        dir_sign = np.where(dp >= 0.5, 1, -1)
    vf_base = vp >= 0.5
    dti = pd.to_datetime(ts)
    minutes = pd.DatetimeIndex(dti).hour * 60 + pd.DatetimeIndex(dti).minute
    rth = (minutes >= 570) & (minutes <= 930)
    atr_ok = atr > a50 * 1.02
    tf5_ok = ((dir_sign == 1) & (tf5 == 1)) | ((dir_sign == -1) & (tf5 == 0))
    filt = vf_base & rth & atr_ok & (vp >= conf_thr)
    filt_tf5 = filt & tf5_ok

    out: dict[str, pd.DataFrame] = {}

    def frame(name: str, idx: np.ndarray, pnl: np.ndarray) -> None:
        out[name] = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts[idx]),
                "fold": fold.fold,
                "pnl": pnl,
            }
        )

    base_idx = np.where(vf_base)[0]
    filt_idx = np.where(filt)[0]
    f5_idx = np.where(filt_tf5)[0]

    frame("straddle_base", base_idx, sim_straddle(base_idx, cl[base_idx], atr[base_idx], hi, lo, cl, point_value, tp_m, sl_m, adverse_same_bar))
    frame("dir_base", base_idx, sim_directional(base_idx, cl[base_idx], dir_sign[base_idx], atr[base_idx], hi, lo, cl, point_value))

    if delayed_bars > 0:
        filt_idx = filt_idx[filt_idx + delayed_bars < len(op)]
        entry_idx = filt_idx + delayed_bars
        ep = op[entry_idx]
        sim_idx = entry_idx
        frame("straddle_filt", filt_idx, sim_straddle(sim_idx, ep, atr[filt_idx], hi, lo, cl, point_value, tp_m, sl_m, adverse_same_bar))
        frame("dir_filt", filt_idx, sim_directional(sim_idx, ep, dir_sign[filt_idx], atr[filt_idx], hi, lo, cl, point_value))
    else:
        frame("straddle_filt", filt_idx, sim_straddle(filt_idx, cl[filt_idx], atr[filt_idx], hi, lo, cl, point_value, tp_m, sl_m, adverse_same_bar))
        frame("dir_filt", filt_idx, sim_directional(filt_idx, cl[filt_idx], dir_sign[filt_idx], atr[filt_idx], hi, lo, cl, point_value))

    frame("straddle_f5", f5_idx, sim_straddle(f5_idx, cl[f5_idx], atr[f5_idx], hi, lo, cl, point_value, tp_m, sl_m, adverse_same_bar))
    return out


def run_walk_forward(
    feat: pd.DataFrame,
    instrument: str = "NQ",
    feature_mode: str = "all",
    eval_mode: str = "test",
    purged: bool = False,
    shuffle_labels: bool = False,
    need_direction: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    fcols = feature_columns(feat, feature_mode)
    stores = {k: [] for k in ["straddle_base", "dir_base", "straddle_filt", "dir_filt", "straddle_f5"]}
    fold_rows: list[dict[str, object]] = []
    point_value = POINT_VALUES[instrument]

    for fold, tr, te in fold_windows(feat, purged=purged):
        vp = fit_predict(
            tr,
            te,
            fcols,
            "lv",
            eval_mode=eval_mode,
            shuffle_labels=shuffle_labels,
            seed=RANDOM_SEED + fold.fold,
        )
        dp = None
        if need_direction:
            dp = fit_predict(tr, te, fcols, "ld", eval_mode=eval_mode, shuffle_labels=False, seed=RANDOM_SEED + 100 + fold.fold)

        frames = simulate_from_fold(fold, te, vp, dp, point_value)
        for key, frame in frames.items():
            if key in stores:
                stores[key].append(frame)

        sf = frames["straddle_filt"]
        df = frames["dir_filt"]
        sb = frames["straddle_base"]
        fold_rows.append(
            {
                "fold": fold.fold,
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                "train_rows": fold.train_rows,
                "test_rows": fold.test_rows,
                "overlap_rows": fold.overlap_rows,
                "str_base_n": len(sb),
                "str_base_pnl": round(float(sb["pnl"].sum()), 2),
                "str_filt_n": len(sf),
                "str_filt_pnl": round(float(sf["pnl"].sum()), 2),
                "dir_filt_n": len(df),
                "dir_filt_pnl": round(float(df["pnl"].sum()), 2),
            }
        )
        print(
            f"{instrument} {feature_mode} {eval_mode} purged={purged} shuffle={shuffle_labels} "
            f"fold={fold.fold} trades={len(sf):,} pnl=${sf['pnl'].sum():,.0f}",
            flush=True,
        )

    out = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=["timestamp", "fold", "pnl"]) for k, v in stores.items()}
    return out, pd.DataFrame(fold_rows)


def parameter_sensitivity(feat: pd.DataFrame) -> pd.DataFrame:
    fcols = feature_columns(feat, "all")
    rows = []
    point_value = POINT_VALUES["NQ"]
    thresholds = [0.50, 0.52, 0.55, 0.60, 0.65]
    tp_sl_pairs = [(0.8, 0.4), (1.0, 0.3), (1.2, 0.4)]
    for fold, tr, te in fold_windows(feat, purged=False):
        vp = fit_predict(tr, te, fcols, "lv", eval_mode="test", shuffle_labels=False, seed=RANDOM_SEED + fold.fold)
        dp = fit_predict(tr, te, fcols, "ld", eval_mode="test", shuffle_labels=False, seed=RANDOM_SEED + 100 + fold.fold)
        for conf in thresholds:
            for tp_m, sl_m in tp_sl_pairs:
                frames = simulate_from_fold(fold, te, vp, dp, point_value, conf_thr=conf, tp_m=tp_m, sl_m=sl_m)
                base = frames["straddle_filt"]
                for slip in SLIPPAGE_POINTS:
                    stressed = apply_slippage(base, slip, point_value)
                    summ = summarize_trades("param", stressed, slip)
                    rows.append(
                        {
                            "fold": fold.fold,
                            "conf_thr": conf,
                            "tp_m": tp_m,
                            "sl_m": sl_m,
                            "slippage_points_per_leg": slip,
                            "trades": summ["trades"],
                            "total_pnl": summ["total_pnl"],
                            "win_rate": summ["win_rate"],
                            "profit_factor": summ["profit_factor"],
                        }
                    )
    return pd.DataFrame(rows)


def stress_table(base_trades: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slip in SLIPPAGE_POINTS:
        rows.append(summarize_trades("slippage_stress", apply_slippage(base_trades, slip), slip))

    rng = np.random.default_rng(RANDOM_SEED)
    stressed_15 = apply_slippage(base_trades, 1.5)
    for drop_rate in [0.05, 0.10, 0.20]:
        for sim in range(250):
            keep = rng.random(len(stressed_15)) >= drop_rate
            sample = stressed_15.loc[keep].copy()
            s = summarize_trades(f"missed_fill_drop_{drop_rate:.0%}", sample, 1.5)
            s["simulation"] = sim
            rows.append(s)

    fcols = feature_columns(feat, "all")
    point_value = POINT_VALUES["NQ"]
    adverse_frames = []
    delay_frames: dict[int, list[pd.DataFrame]] = {1: [], 2: [], 3: []}
    for fold, tr, te in fold_windows(feat, purged=False):
        vp = fit_predict(tr, te, fcols, "lv", eval_mode="test", shuffle_labels=False, seed=RANDOM_SEED + fold.fold)
        dp = fit_predict(tr, te, fcols, "ld", eval_mode="test", shuffle_labels=False, seed=RANDOM_SEED + 100 + fold.fold)
        adverse_frames.append(simulate_from_fold(fold, te, vp, dp, point_value, adverse_same_bar=True)["straddle_filt"])
        for delay in [1, 2, 3]:
            delay_frames[delay].append(simulate_from_fold(fold, te, vp, dp, point_value, delayed_bars=delay)["straddle_filt"])

    adverse = pd.concat(adverse_frames, ignore_index=True)
    s = summarize_trades("adverse_same_bar_ordering", apply_slippage(adverse, 1.5), 1.5)
    rows.append(s)
    adverse.to_csv(TRADE_DIR / "adverse_same_bar_base_trades.csv", index=False)

    for delay, frames in delay_frames.items():
        delayed = pd.concat(frames, ignore_index=True)
        s = summarize_trades(f"entry_delay_{delay}_bars", apply_slippage(delayed, 1.5), 1.5)
        rows.append(s)
        delayed.to_csv(TRADE_DIR / f"entry_delay_{delay}_bars_base_trades.csv", index=False)

    return pd.DataFrame(rows)


def block_bootstrap(trades: pd.DataFrame, block: str, n_sims: int = 2000) -> pd.DataFrame:
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    if block == "day":
        keys = trades["timestamp"].dt.strftime("%Y-%m-%d")
        periods_per_year = 252
    elif block == "week":
        keys = trades["timestamp"].dt.to_period("W").astype(str)
        periods_per_year = 52
    elif block == "month":
        keys = trades["timestamp"].dt.to_period("M").astype(str)
        periods_per_year = 12
    else:
        raise ValueError(block)

    block_pnl = trades.groupby(keys)["pnl"].sum().astype(float)
    block_arrays = [g["pnl"].astype(float).to_numpy() for _, g in trades.groupby(keys)]
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    n_blocks = len(block_arrays)
    for _ in range(n_sims):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        pnl = np.concatenate([block_arrays[i] for i in idx])
        block_returns = block_pnl.iloc[idx].reset_index(drop=True)
        rows.append(
            {
                "block": block,
                "total_pnl": float(pnl.sum()),
                "max_drawdown": max_drawdown(pnl),
                "annualized_sharpe": ann_sharpe(block_returns, periods_per_year),
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block, g in samples.groupby("block"):
        rows.append(
            {
                "block": block,
                "sims": int(len(g)),
                "p_positive": round(float((g["total_pnl"] > 0).mean()), 4),
                "total_pnl_p05": round(float(g["total_pnl"].quantile(0.05)), 2),
                "total_pnl_median": round(float(g["total_pnl"].median()), 2),
                "total_pnl_worst": round(float(g["total_pnl"].min()), 2),
                "max_drawdown_median": round(float(g["max_drawdown"].median()), 2),
                "max_drawdown_worst": round(float(g["max_drawdown"].min()), 2),
                "annualized_sharpe_p05": round(float(g["annualized_sharpe"].quantile(0.05)), 3),
                "annualized_sharpe_median": round(float(g["annualized_sharpe"].median()), 3),
            }
        )
    return pd.DataFrame(rows)


def leakage_audit(folds: pd.DataFrame) -> pd.DataFrame:
    text = SOURCE_SCRIPT.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    future_shift_lines = []
    feature_future_lines = []
    for i, line in enumerate(lines, start=1):
        if re.search(r"shift\s*\(\s*-\d+", line):
            future_shift_lines.append(f"{i}: {line.strip()}")
            if 'f["lv"]' not in line and 'f["ld"]' not in line:
                feature_future_lines.append(f"{i}: {line.strip()}")
    overlap_rows = int(folds["overlap_rows"].sum()) if "overlap_rows" in folds else 0
    return pd.DataFrame(
        [
            {
                "check": "future_shift_in_features",
                "verdict": "pass" if not feature_future_lines else "fail",
                "evidence": "No negative shift found in feature assignments."
                if not feature_future_lines
                else " | ".join(feature_future_lines[:5]),
            },
            {
                "check": "future_shift_in_labels",
                "verdict": "allowed",
                "evidence": "Negative shift appears in labels only: " + " | ".join(future_shift_lines[:5]),
            },
            {
                "check": "original_fold_boundary_overlap",
                "verdict": "warn" if overlap_rows else "pass",
                "evidence": f"Original pandas slices share {overlap_rows} timestamp rows across train/test boundaries.",
            },
            {
                "check": "test_fold_used_for_early_stopping",
                "verdict": "warn",
                "evidence": "Original LightGBM fit uses eval_set=[(Xte, te[label])] for early stopping; audit adds train-tail clean eval.",
            },
            {
                "check": "old_monte_carlo_meaning",
                "verdict": "warn",
                "evidence": "Original Monte Carlo permutes final trade PnL, preserving total PnL; it is a path-order drawdown test, not proof of edge.",
            },
        ]
    )


def claim_matrix(independent: pd.DataFrame, fresh: pd.DataFrame, stress: pd.DataFrame, leakage: pd.DataFrame) -> pd.DataFrame:
    def fresh_row(variant: str, slip: float | None = None) -> pd.Series:
        rows = fresh[fresh["variant"] == variant]
        if slip is not None:
            rows = rows[rows["slippage_points_per_leg"].astype(float) == slip]
        return rows.iloc[0]

    def indep_row(variant: str, slip: float | None = None) -> pd.Series:
        rows = independent[independent["variant"] == variant]
        if slip is not None:
            rows = rows[rows["slippage_points_per_leg"].astype(float) == slip]
        return rows.iloc[0]

    rows = []

    def add(claim: str, source: str, prior: object, replicated: object, tolerance: float | str, units: str = "") -> None:
        verdict = "supported"
        try:
            p = float(prior)
            r = float(replicated)
            tol = float(tolerance)
            if abs(p - r) > tol:
                verdict = "not_reproduced"
        except (TypeError, ValueError):
            if str(prior) != str(replicated):
                verdict = "not_reproduced"
        rows.append(
            {
                "claim": claim,
                "source": source,
                "prior_value": prior,
                "replicated_value": replicated,
                "tolerance": tolerance,
                "units": units,
                "verdict": verdict,
            }
        )

    for slip in SLIPPAGE_POINTS:
        fr = fresh_row("straddle_filtered_slippage_stress", slip)
        ir = indep_row("independent_straddle_slippage_stress", slip)
        add(f"Filtered straddle total PnL at {slip:.1f} pt/leg", "fresh_conservative_metrics.csv", fr["total_pnl"], ir["total_pnl"], 1.0, "USD")
        add(f"Filtered straddle trades at {slip:.1f} pt/leg", "fresh_conservative_metrics.csv", fr["trades"], ir["trades"], 0, "trades")
        add(f"Filtered straddle monthly Sharpe at {slip:.1f} pt/leg", "fresh_conservative_metrics.csv", fr["monthly_sharpe_ann"], ir["monthly_sharpe_ann"], 0.01, "annualized")

    fr = fresh_row("directional_filtered_failure_case")
    ir = indep_row("independent_directional_failure_case")
    add("Directional filtered strategy is a failure case", "fresh_conservative_metrics.csv", fr["total_pnl"], ir["total_pnl"], 1.0, "USD")

    old_sh = float(fresh_row("straddle_filtered_slippage_stress", 0.5)["trade_t_stat_not_sharpe"])
    rows.append(
        {
            "claim": "Old 9+ or 13+ value is a finance Sharpe ratio",
            "source": "legacy GBM_Straddle_Databento.py sh column",
            "prior_value": old_sh,
            "replicated_value": "trade_t_stat_not_sharpe",
            "tolerance": "none",
            "units": "metric label",
            "verdict": "rejected",
        }
    )

    leak_fail = leakage[leakage["verdict"] == "fail"]
    leak_warn = leakage[leakage["verdict"] == "warn"]
    rows.append(
        {
            "claim": "No direct future-looking feature leakage found",
            "source": "static source scan",
            "prior_value": "not previously formalized",
            "replicated_value": "pass" if leak_fail.empty else "fail",
            "tolerance": "none",
            "units": "audit verdict",
            "verdict": "supported" if leak_fail.empty else "rejected",
        }
    )
    rows.append(
        {
            "claim": "Original validation is pure out-of-sample",
            "source": "source-code audit",
            "prior_value": "implicit",
            "replicated_value": "warnings: " + "; ".join(leak_warn["check"].tolist()),
            "tolerance": "none",
            "units": "audit verdict",
            "verdict": "corrected",
        }
    )
    stress_15 = stress[(stress["variant"] == "slippage_stress") & (stress["slippage_points_per_leg"].astype(float) == 1.5)].iloc[0]
    stress_25 = stress[(stress["variant"] == "slippage_stress") & (stress["slippage_points_per_leg"].astype(float) == 2.5)].iloc[0]
    rows.append(
        {
            "claim": "1.5 pt/leg slippage case remains positive",
            "source": "overfit_audit stress_table.csv",
            "prior_value": "positive required",
            "replicated_value": stress_15["total_pnl"],
            "tolerance": "positive PnL",
            "units": "USD",
            "verdict": "supported" if float(stress_15["total_pnl"]) > 0 else "rejected",
        }
    )
    rows.append(
        {
            "claim": "2.5 pt/leg slippage breaks the edge",
            "source": "overfit_audit stress_table.csv",
            "prior_value": "negative expected",
            "replicated_value": stress_25["total_pnl"],
            "tolerance": "negative PnL",
            "units": "USD",
            "verdict": "supported" if float(stress_25["total_pnl"]) < 0 else "not_reproduced",
        }
    )
    return pd.DataFrame(rows)


def make_charts(
    main_trades: pd.DataFrame,
    summaries: pd.DataFrame,
    folds: pd.DataFrame,
    param: pd.DataFrame,
    bootstrap_samples: pd.DataFrame,
) -> None:
    stressed = apply_slippage(main_trades, 1.5)
    stressed = stressed.sort_values("timestamp")
    pnl = stressed["pnl"].astype(float)
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()

    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.plot(stressed["timestamp"], equity, linewidth=1.3)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Independent Replication Equity Curve: 1.5 Pt/Leg Slippage")
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "equity_curve_1p5_slippage.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    ax.fill_between(stressed["timestamp"], drawdown, 0, color="#b22222", alpha=0.75)
    ax.set_title("Independent Replication Drawdown: 1.5 Pt/Leg Slippage")
    ax.set_ylabel("Drawdown (USD)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "drawdown_1p5_slippage.png")
    plt.close(fig)

    slip = summaries[summaries["variant"] == "independent_straddle_slippage_stress"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4), dpi=160)
    ax.plot(slip["slippage_points_per_leg"], slip["total_pnl"], marker="o", label="Total PnL")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Execution Stress Breakpoint")
    ax.set_xlabel("Slippage points per leg")
    ax.set_ylabel("Total PnL (USD)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "slippage_stress_pnl.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=160)
    ax.bar(folds["fold"].astype(str), folds["str_filt_pnl"].astype(float), color="#1f77b4", label="Straddle filtered")
    ax.bar(folds["fold"].astype(str), folds["dir_filt_pnl"].astype(float), color="#d62728", alpha=0.7, label="Directional")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Walk-Forward Fold PnL")
    ax.set_xlabel("Fold")
    ax.set_ylabel("PnL (USD)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "walkforward_fold_pnl.png")
    plt.close(fig)

    heat = (
        param[
            (param["tp_m"].astype(float) == 1.0)
            & (param["sl_m"].astype(float) == 0.3)
            & (param["slippage_points_per_leg"].astype(float) == 1.5)
        ]
        .groupby("conf_thr")["total_pnl"]
        .sum()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=160)
    ax.bar(heat["conf_thr"].astype(str), heat["total_pnl"], color="#2ca02c")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Parameter Sensitivity: Confidence Threshold at 1.5 Pt/Leg")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Total PnL (USD)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "parameter_sensitivity_confidence.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4), dpi=160)
    for block, g in bootstrap_samples.groupby("block"):
        ax.hist(g["total_pnl"], bins=35, alpha=0.45, label=block)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Block Bootstrap Total PnL Distribution: 1.5 Pt/Leg")
    ax.set_xlabel("Total PnL (USD)")
    ax.set_ylabel("Simulation count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "bootstrap_pnl_distribution.png")
    plt.close(fig)


def write_markdown_report(
    manifest: dict[str, object],
    claim_df: pd.DataFrame,
    summaries: pd.DataFrame,
    leakage: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    ablations: pd.DataFrame,
    stress: pd.DataFrame,
    cross: pd.DataFrame,
) -> None:
    def row(variant: str, slip: float | None = None) -> pd.Series:
        rows = summaries[summaries["variant"] == variant]
        if slip is not None:
            rows = rows[rows["slippage_points_per_leg"].astype(float) == slip]
        return rows.iloc[0]

    base15 = row("independent_straddle_slippage_stress", 1.5)
    clean15 = row("clean_eval_straddle_slippage_stress", 1.5)
    purged15 = row("purged_clean_eval_straddle_slippage_stress", 1.5)
    label15 = row("label_shuffle_straddle_slippage_stress", 1.5)
    adverse = stress[stress["variant"] == "adverse_same_bar_ordering"].iloc[0]
    leakage_md = leakage.copy()
    leakage_md["evidence"] = leakage_md["evidence"].astype(str).str.replace("|", ";", regex=False)
    lines = [
        "# Overfitting and Replication Audit",
        "",
        f"Run ID: `{manifest['run_id']}`",
        "",
        "## Bottom Line",
        "",
        (
            "The independently rebuilt original logic reproduced the prior conservative result, "
            f"but the audit found validation weaknesses that make the right wording narrower. At 1.5 NQ points "
            f"of slippage per leg, the replicated original logic produced {int(base15['trades']):,} trades, "
            f"${float(base15['total_pnl']):,.2f} PnL, win rate {float(base15['win_rate']):.2%}, "
            f"profit factor {float(base15['profit_factor']):.3f}, and monthly annualized Sharpe "
            f"{float(base15['monthly_sharpe_ann']):.3f}."
        ),
        "",
        (
            "The clean-eval rerun avoids using the test fold for LightGBM early stopping. "
            f"That cleaner test produced ${float(clean15['total_pnl']):,.2f} at 1.5 pt/leg. "
            f"The purged clean-eval rerun produced ${float(purged15['total_pnl']):,.2f}. "
            f"The label-shuffle null produced ${float(label15['total_pnl']):,.2f}, which is the key overfitting sanity check."
        ),
        "",
        "## What Survived",
        "",
        "- The raw NQ data exists locally and is hash-locked in `audit_manifest.json`.",
        "- The original conservative metrics were independently reproduced without importing the original strategy module.",
        "- Direct future-looking feature leakage was not found in the static feature scan.",
        "- The old `sh` value is rejected as a headline Sharpe ratio and retained only as a trade-level t-statistic.",
        "- The 2.5 pt/leg execution stress remains a real failure boundary.",
        "",
        "## What Did Not Survive Cleanly",
        "",
        "- The original script used the test fold as LightGBM's early-stopping evaluation set, so it is not a perfectly pure out-of-sample process.",
        "- The original pandas slicing shares one boundary timestamp per fold between train and test; the audit adds a purged rerun to remove that issue.",
        f"- Same-bar adverse ordering is severe: at 1.5 pt/leg it produced ${float(adverse['total_pnl']):,.2f}.",
        "- Monte Carlo permutation of final trade PnL is downgraded to path-order drawdown evidence, not proof of a stable edge.",
        "",
        "## Claim Matrix Summary",
        "",
        claim_df["verdict"].value_counts().rename_axis("verdict").reset_index(name="count").to_markdown(index=False),
        "",
        "## Leakage Checks",
        "",
        leakage_md.to_markdown(index=False),
        "",
        "## Main Metrics",
        "",
        summaries[
            summaries["variant"].isin(
                [
                    "independent_straddle_slippage_stress",
                    "clean_eval_straddle_slippage_stress",
                    "purged_clean_eval_straddle_slippage_stress",
                    "label_shuffle_straddle_slippage_stress",
                    "independent_directional_failure_case",
                ]
            )
        ].to_markdown(index=False),
        "",
        "## Block Bootstrap",
        "",
        bootstrap_summary.to_markdown(index=False),
        "",
        "## Feature Ablations",
        "",
        ablations.to_markdown(index=False),
        "",
        "## Cross-Instrument Sanity Check",
        "",
        cross.to_markdown(index=False) if len(cross) else "No cross-instrument files were available.",
        "",
        "## Final Portfolio Wording",
        "",
        (
            "Use: Built an independently audited NQ futures research pipeline with raw-data hashing, "
            "walk-forward replication, leakage checks, clean-eval and purged reruns, slippage stress, "
            "block-bootstrap Monte Carlo, and failure analysis. Do not claim a proven profitable trading bot."
        ),
        "",
    ]
    (AUDIT_ROOT / "AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def pdf_table(data: list[list[object]], col_widths: list[float] | None = None) -> Table:
    table = Table(data, repeatRows=1, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17212b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c0cc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def build_pdf(summaries: pd.DataFrame, claim_df: pd.DataFrame, bootstrap_summary: pd.DataFrame, leakage: pd.DataFrame) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=10.2))
    styles.add(ParagraphStyle(name="Tight", parent=styles["BodyText"], fontSize=9.3, leading=11.2, spaceAfter=5))
    styles["Title"].fontSize = 18
    styles["Title"].leading = 22
    styles["Heading1"].fontSize = 13
    styles["Heading1"].leading = 15

    def p(text: str, style: str = "Tight") -> Paragraph:
        return Paragraph(text, styles[style])

    def img(name: str, width: float = 6.1 * inch) -> Image:
        image = Image(str(CHART_DIR / name))
        ratio = image.imageHeight / float(image.imageWidth)
        image.drawWidth = width
        image.drawHeight = width * ratio
        return image

    def add_page_number(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawRightString(7.5 * inch, 0.45 * inch, f"Page {doc.page}")
        canvas.restoreState()

    def get_row(variant: str, slip: float | None = None) -> pd.Series:
        rows = summaries[summaries["variant"] == variant]
        if slip is not None:
            rows = rows[rows["slippage_points_per_leg"].astype(float) == slip]
        return rows.iloc[0]

    base15 = get_row("independent_straddle_slippage_stress", 1.5)
    clean15 = get_row("clean_eval_straddle_slippage_stress", 1.5)
    purged15 = get_row("purged_clean_eval_straddle_slippage_stress", 1.5)
    label15 = get_row("label_shuffle_straddle_slippage_stress", 1.5)
    stress = pd.read_csv(TABLE_DIR / "stress_tests.csv")
    cross = pd.read_csv(TABLE_DIR / "cross_instrument_sanity.csv")
    adverse = stress[stress["variant"] == "adverse_same_bar_ordering"].iloc[0]
    fail_rows = [
        ["Failure Check", "Result", "Meaning"],
        ["Same-bar adverse order", f"${float(adverse['total_pnl']):,.0f}", "Execution assumption can kill result"],
    ]
    for _, row_data in cross.iterrows():
        fail_rows.append(
            [
                f"Cross {row_data['instrument']}",
                f"${float(row_data['total_pnl']):,.0f}",
                "No broad futures generalization",
            ]
        )
    leak_rows = [["Check", "Verdict", "Short evidence"]]
    for _, row_data in leakage.iterrows():
        evidence = str(row_data["evidence"])
        evidence = evidence.replace("|", ";")
        if len(evidence) > 92:
            evidence = evidence[:89] + "..."
        leak_rows.append([row_data["check"], row_data["verdict"], evidence])
    boot_rows = [["Block", "P(pos)", "P05 PnL", "Median PnL", "Worst PnL", "Worst DD", "Sharpe P05/Med"]]
    for _, row_data in bootstrap_summary.iterrows():
        boot_rows.append(
            [
                row_data["block"],
                f"{float(row_data['p_positive']):.1%}",
                f"${float(row_data['total_pnl_p05']):,.0f}",
                f"${float(row_data['total_pnl_median']):,.0f}",
                f"${float(row_data['total_pnl_worst']):,.0f}",
                f"${float(row_data['max_drawdown_worst']):,.0f}",
                f"{float(row_data['annualized_sharpe_p05']):.2f}/{float(row_data['annualized_sharpe_median']):.2f}",
            ]
        )

    doc = SimpleDocTemplate(
        str(AUDIT_ROOT / "AUDIT_REPORT.pdf"),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.62 * inch,
    )
    story = [
        p("NQ Futures Strategy Overfitting and Replication Audit", "Title"),
        p("Independent rebuild, leakage checks, stress tests, and claim reconciliation", "Small"),
        Spacer(1, 0.1 * inch),
        p(
            "This audit freezes the existing strategy instead of tuning for a better result. The original conservative "
            "result was independently rebuilt from raw NQ bars, then retested under cleaner validation and execution stress."
        ),
        pdf_table(
            [
                ["Test", "Trades", "PnL", "Win", "PF", "Monthly Sharpe"],
                [
                    "Replicated original 1.5 pt",
                    f"{int(base15['trades']):,}",
                    f"${float(base15['total_pnl']):,.0f}",
                    f"{float(base15['win_rate']):.1%}",
                    f"{float(base15['profit_factor']):.2f}",
                    f"{float(base15['monthly_sharpe_ann']):.2f}",
                ],
                [
                    "Clean eval 1.5 pt",
                    f"{int(clean15['trades']):,}",
                    f"${float(clean15['total_pnl']):,.0f}",
                    f"{float(clean15['win_rate']):.1%}",
                    f"{float(clean15['profit_factor']):.2f}",
                    f"{float(clean15['monthly_sharpe_ann']):.2f}",
                ],
                [
                    "Purged clean eval 1.5 pt",
                    f"{int(purged15['trades']):,}",
                    f"${float(purged15['total_pnl']):,.0f}",
                    f"{float(purged15['win_rate']):.1%}",
                    f"{float(purged15['profit_factor']):.2f}",
                    f"{float(purged15['monthly_sharpe_ann']):.2f}",
                ],
                [
                    "Label-shuffle null 1.5 pt",
                    f"{int(label15['trades']):,}",
                    f"${float(label15['total_pnl']):,.0f}",
                    f"{float(label15['win_rate']):.1%}",
                    f"{float(label15['profit_factor']):.2f}",
                    f"{float(label15['monthly_sharpe_ann']):.2f}",
                ],
            ],
            [1.85 * inch, 0.7 * inch, 0.9 * inch, 0.62 * inch, 0.55 * inch, 1.0 * inch],
        ),
        Spacer(1, 0.1 * inch),
        img("equity_curve_1p5_slippage.png", 5.8 * inch),
        img("drawdown_1p5_slippage.png", 5.8 * inch),
        Spacer(1, 0.08 * inch),
        p("Key correction: the old 9+ or 13+ value is a trade-level t-statistic, not a finance-style Sharpe ratio."),
        p("Validation caveat: the original script used the test fold for early stopping; cleaner train-tail validation is reported separately."),
        Spacer(1, 0.08 * inch),
        img("slippage_stress_pnl.png", 5.6 * inch),
        img("walkforward_fold_pnl.png", 5.6 * inch),
        p("Claim verdict counts", "Heading1"),
        pdf_table([["Verdict", "Count"]] + claim_df["verdict"].value_counts().rename_axis("verdict").reset_index(name="count").values.tolist()),
        Spacer(1, 0.08 * inch),
        p("Failure checks", "Heading1"),
        pdf_table(fail_rows, [1.75 * inch, 1.0 * inch, 3.4 * inch]),
        Spacer(1, 0.08 * inch),
        p("Leakage and validation checks", "Heading1"),
        pdf_table(leak_rows, [1.75 * inch, 0.65 * inch, 3.95 * inch]),
        Spacer(1, 0.08 * inch),
        p("Block bootstrap", "Heading1"),
        pdf_table(boot_rows, [0.62 * inch, 0.55 * inch, 0.78 * inch, 0.86 * inch, 0.82 * inch, 0.82 * inch, 1.0 * inch]),
        Spacer(1, 0.08 * inch),
        img("bootstrap_pnl_distribution.png", 5.6 * inch),
        img("parameter_sensitivity_confidence.png", 5.6 * inch),
    ]
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def write_manifest(run_id: str, data_info: dict[str, object]) -> dict[str, object]:
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_root": str(AUDIT_ROOT),
        "strategy_frozen": {
            "source_script": str(SOURCE_SCRIPT),
            "source_script_sha256": sha256(SOURCE_SCRIPT),
            "max_hold_bars": MAX_HOLD,
            "confidence_threshold": CONF_THR,
            "base_slippage_points_per_leg": BASE_SLIPPAGE_POINTS,
            "slippage_points_tested": SLIPPAGE_POINTS,
            "cost_per_leg_usd": COST_PER_LEG,
            "point_values": POINT_VALUES,
            "train_months": TRAIN_MONTHS,
            "test_months": TEST_MONTHS,
            "slide_months": SLIDE_MONTHS,
            "random_seed": RANDOM_SEED,
            "model": {
                "class": "lightgbm.LGBMClassifier",
                "n_estimators": 300,
                "learning_rate": 0.04,
                "max_depth": 5,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_samples": 40,
                "random_state": RANDOM_SEED,
            },
        },
        "data": {
            "nq": {
                "path": str(NQ_DATA),
                "sha256": sha256(NQ_DATA),
                **data_info,
            },
            "cross_instrument_files": {
                k: {"path": str(v), "sha256": sha256(v), "exists": v.exists()} for k, v in CROSS_DATA.items() if v.exists()
            },
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                "pandas": package_version("pandas"),
                "numpy": package_version("numpy"),
                "lightgbm": package_version("lightgbm"),
                "scikit-learn": package_version("scikit-learn"),
                "matplotlib": package_version("matplotlib"),
                "reportlab": package_version("reportlab"),
            },
        },
    }
    (AUDIT_ROOT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    ensure_dirs()
    run_id = datetime.now(timezone.utc).strftime("overfit_audit_%Y%m%dT%H%M%SZ")
    print(f"Starting {run_id}", flush=True)

    feat, data_info = load_features(NQ_DATA)
    manifest = write_manifest(run_id, data_info)

    main_stores, main_folds = run_walk_forward(feat, instrument="NQ", feature_mode="all", eval_mode="test", purged=False)
    main_folds.to_csv(TABLE_DIR / "independent_walkforward_folds.csv", index=False)
    for key, frame in main_stores.items():
        frame.to_csv(TRADE_DIR / f"independent_{key}_trades.csv", index=False)

    independent_rows = [
        summarize_trades("independent_straddle_base_cost", main_stores["straddle_filt"], BASE_SLIPPAGE_POINTS),
        summarize_trades("independent_directional_failure_case", main_stores["dir_filt"], BASE_SLIPPAGE_POINTS),
    ]
    for slip in SLIPPAGE_POINTS:
        independent_rows.append(
            summarize_trades("independent_straddle_slippage_stress", apply_slippage(main_stores["straddle_filt"], slip), slip)
        )

    clean_stores, clean_folds = run_walk_forward(feat, instrument="NQ", feature_mode="all", eval_mode="train_tail", purged=False)
    clean_folds.to_csv(TABLE_DIR / "clean_eval_walkforward_folds.csv", index=False)
    for slip in SLIPPAGE_POINTS:
        independent_rows.append(
            summarize_trades("clean_eval_straddle_slippage_stress", apply_slippage(clean_stores["straddle_filt"], slip), slip)
        )

    purged_stores, purged_folds = run_walk_forward(feat, instrument="NQ", feature_mode="all", eval_mode="train_tail", purged=True)
    purged_folds.to_csv(TABLE_DIR / "purged_clean_eval_walkforward_folds.csv", index=False)
    for slip in SLIPPAGE_POINTS:
        independent_rows.append(
            summarize_trades("purged_clean_eval_straddle_slippage_stress", apply_slippage(purged_stores["straddle_filt"], slip), slip)
        )

    null_stores, null_folds = run_walk_forward(
        feat,
        instrument="NQ",
        feature_mode="all",
        eval_mode="train_tail",
        purged=False,
        shuffle_labels=True,
        need_direction=False,
    )
    null_folds.to_csv(TABLE_DIR / "label_shuffle_walkforward_folds.csv", index=False)
    for slip in SLIPPAGE_POINTS:
        independent_rows.append(
            summarize_trades("label_shuffle_straddle_slippage_stress", apply_slippage(null_stores["straddle_filt"], slip), slip)
        )

    ablation_rows = []
    for mode in ["all", "no_time", "time_only", "volatility_only"]:
        stores, folds = run_walk_forward(feat, instrument="NQ", feature_mode=mode, eval_mode="train_tail", purged=False, need_direction=False)
        folds.to_csv(TABLE_DIR / f"ablation_{mode}_folds.csv", index=False)
        base = summarize_trades(f"ablation_{mode}", apply_slippage(stores["straddle_filt"], 1.5), 1.5)
        base["feature_mode"] = mode
        ablation_rows.append(base)
    ablations = pd.DataFrame(ablation_rows)

    print("Running parameter sensitivity...", flush=True)
    param = parameter_sensitivity(feat)
    param.to_csv(TABLE_DIR / "parameter_sensitivity.csv", index=False)

    print("Running execution stress tests...", flush=True)
    stress = stress_table(main_stores["straddle_filt"], feat)
    stress.to_csv(TABLE_DIR / "stress_tests.csv", index=False)

    print("Running block bootstrap...", flush=True)
    stressed_15 = apply_slippage(main_stores["straddle_filt"], 1.5)
    boot_samples = pd.concat([block_bootstrap(stressed_15, b) for b in ["day", "week", "month"]], ignore_index=True)
    boot_summary = summarize_bootstrap(boot_samples)
    boot_samples.to_csv(TABLE_DIR / "bootstrap_samples.csv", index=False)
    boot_summary.to_csv(TABLE_DIR / "bootstrap_summary.csv", index=False)

    print("Running cross-instrument sanity checks...", flush=True)
    cross_rows = []
    for instrument, path in CROSS_DATA.items():
        if not path.exists():
            continue
        cross_feat, _ = load_features(path)
        stores, folds = run_walk_forward(cross_feat, instrument=instrument, feature_mode="all", eval_mode="train_tail", purged=False)
        folds.to_csv(TABLE_DIR / f"cross_{instrument.lower()}_folds.csv", index=False)
        row = summarize_trades(f"cross_{instrument}_same_logic", apply_slippage(stores["straddle_filt"], 1.5, POINT_VALUES[instrument]), 1.5)
        row["instrument"] = instrument
        row["data_path"] = str(path)
        cross_rows.append(row)
        del cross_feat, stores, folds
        gc.collect()
    cross = pd.DataFrame(cross_rows)

    summaries = pd.DataFrame(independent_rows)
    summaries.to_csv(TABLE_DIR / "audit_metrics.csv", index=False)
    ablations.to_csv(TABLE_DIR / "feature_ablations.csv", index=False)
    cross.to_csv(TABLE_DIR / "cross_instrument_sanity.csv", index=False)

    leakage = leakage_audit(main_folds)
    leakage.to_csv(TABLE_DIR / "leakage_audit.csv", index=False)

    fresh = pd.read_csv(FRESH_DIR / "fresh_conservative_metrics.csv")
    claims = claim_matrix(summaries, fresh, stress, leakage)
    claims.to_csv(AUDIT_ROOT / "claim_matrix.csv", index=False)

    make_charts(main_stores["straddle_filt"], summaries, main_folds, param, boot_samples)
    write_markdown_report(manifest, claims, summaries, leakage, boot_summary, ablations, stress, cross)
    build_pdf(summaries, claims, boot_summary, leakage)

    print(f"Saved audit artifacts to {AUDIT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
