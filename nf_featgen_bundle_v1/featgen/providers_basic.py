from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .feature_naming import prefixed

# ---- helpers ----

def _ensure_ts(df: pd.DataFrame, id_col: str, ds_col: str, y_col: str) -> pd.DataFrame:
    cols = [id_col, ds_col, y_col]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"必要列が見つかりません: {c}")
    out = df[[id_col, ds_col, y_col]].copy()
    out[ds_col] = pd.to_datetime(out[ds_col], utc=False, errors="coerce")
    if out[ds_col].isna().any():
        raise ValueError("dsにNaTが含まれています。データを確認してください。")
    return out.sort_values([id_col, ds_col]).reset_index(drop=True)

def infer_freq_safe(s: pd.Series) -> str:
    # try per-series frequency inference; fallback to 'D'
    try:
        f = pd.infer_freq(s)
        if f is None:
            return "D"
        return f
    except Exception:
        return "D"

# ---- FUTR from ds ----

def make_futr_from_ds(index_df: pd.DataFrame, cfg: Dict, id_col: str, ds_col: str) -> pd.DataFrame:
    # index_df: ['unique_id','ds'] for both hist/futr periods
    df = index_df[[id_col, ds_col]].copy()
    ds = df[ds_col]
    df[prefixed("year", "futr")] = ds.dt.year
    df[prefixed("quarter", "futr")] = ds.dt.quarter
    df[prefixed("month", "futr")] = ds.dt.month
    df[prefixed("weekofyear", "futr")] = ds.dt.isocalendar().week.astype(int)
    df[prefixed("day", "futr")] = ds.dt.day
    df[prefixed("dayofweek", "futr")] = ds.dt.dayofweek
    df[prefixed("is_month_start", "futr")] = ds.dt.is_month_start.astype("int8")
    df[prefixed("is_month_end", "futr")] = ds.dt.is_month_end.astype("int8")
    df[prefixed("is_quarter_start", "futr")] = ds.dt.is_quarter_start.astype("int8")
    df[prefixed("is_quarter_end", "futr")] = ds.dt.is_quarter_end.astype("int8")
    df[prefixed("is_weekend", "futr")] = ds.dt.dayofweek.isin([5,6]).astype("int8")
    # Fourier (optional)
    fourier = cfg.get("fourier", [])
    if fourier:
        for spec in fourier:
            P = int(spec.get("period", 7))
            k = int(spec.get("k", 3))
            t = (ds.view("int64") // 10**9).astype(float)  # seconds since epoch
            # Normalize t per series start to avoid huge phases
            # We'll use ordinal day to stabilize
            t = ds.map(lambda x: x.toordinal()).astype(float)
            for i in range(1, k+1):
                df[prefixed(f"fourier_P{P}_s{i}", "futr")] = np.sin(2*np.pi*i*(t % P)/P)
                df[prefixed(f"fourier_P{P}_c{i}", "futr")] = np.cos(2*np.pi*i*(t % P)/P)
    return df

# ---- HIST from y (lags/roll/diff) ----

def make_hist_from_y(train_df: pd.DataFrame, cfg: Dict, id_col: str, ds_col: str, y_col: str) -> pd.DataFrame:
    df = _ensure_ts(train_df, id_col, ds_col, y_col)
    out = df[[id_col, ds_col]].copy()

    lags: List[int] = cfg.get("lags", [1, 7, 14, 28])
    for L in sorted(set(lags)):
        out[prefixed(f"y_lag{L}", "hist")] = df.groupby(id_col)[y_col].shift(L)

    for wspec in cfg.get("roll", [{"window":7,"stats":["mean","std"]}]):
        w = int(wspec.get("window", 7))
        stats = wspec.get("stats", ["mean", "std"])
        g = df.groupby(id_col)[y_col]
        if "mean" in stats:
            out[prefixed(f"y_roll_mean_{w}", "hist")] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=max(1, w//3)).mean())
        if "std" in stats:
            out[prefixed(f"y_roll_std_{w}", "hist")] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=max(1, w//3)).std())
        if "min" in stats:
            out[prefixed(f"y_roll_min_{w}", "hist")] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=max(1, w//3)).min())
        if "max" in stats:
            out[prefixed(f"y_roll_max_{w}", "hist")] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=max(1, w//3)).max())
        if "median" in stats:
            out[prefixed(f"y_roll_median_{w}", "hist")] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=max(1, w//3)).median())

    diffs: List[int] = cfg.get("diffs", [1])
    for d in sorted(set(diffs)):
        out[prefixed(f"y_diff_{d}", "hist")] = df.groupby(id_col)[y_col].diff(d)

    seas = cfg.get("seasonal_diff", None)
    if seas:
        P = int(seas)
        out[prefixed(f"y_seasonal_diff_{P}", "hist")] = df.groupby(id_col)[y_col].diff(P)

    # Optionally include pct_change
    pct_k = cfg.get("pct_change", [])
    for k in pct_k:
        out[prefixed(f"y_pct_change_{k}", "hist")] = df.groupby(id_col)[y_col].pct_change(k)

    return out

# ---- HIST anomaly: Hampel ----

def _rolling_median(s: pd.Series, w: int):
    return s.rolling(w, min_periods=max(3, w//4)).median()

def _rolling_mad(s: pd.Series, w: int):
    med = _rolling_median(s, w)
    return (s - med).abs().rolling(w, min_periods=max(3, w//4)).median()

def make_hist_anomaly_hampel(train_df: pd.DataFrame, cfg: Dict, id_col: str, ds_col: str, y_col: str) -> pd.DataFrame:
    # cfg: {"windows":[28], "tau":3.5}
    df = _ensure_ts(train_df, id_col, ds_col, y_col)
    out = df[[id_col, ds_col]].copy()
    wins = cfg.get("windows", [28])
    tau = float(cfg.get("tau", 3.5))
    K = 1.4826  # consistency for normal distribution
    g = df.groupby(id_col)[y_col]
    for w in wins:
        med = g.transform(lambda s: _rolling_median(s.shift(1), w))
        mad = g.transform(lambda s: _rolling_mad(s.shift(1), w))
        z = (df[y_col] - med) / (K * mad.replace(0, np.nan))
        out[prefixed(f"y_anom_hampel_z_w{w}", "hist")] = z.astype("float32")
        out[prefixed(f"y_anom_hampel_flag_w{w}", "hist")] = (z.abs() > tau).astype("int8")
        # 有効フラグ（十分な履歴があるか）
        valid = g.transform(lambda s: s.shift(1).rolling(w, min_periods=max(3, w//4)).count()) >= max(3, w//4)
        out[prefixed(f"valid_w{w}", "hist")] = valid.astype("int8")
    return out
