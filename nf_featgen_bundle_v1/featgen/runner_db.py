from __future__ import annotations
import pandas as pd
import numpy as np
import yaml
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from sqlalchemy import text
from .db_utils import get_engine
from db_config import TABLE_PREFIX
from .providers_basic import (
    make_futr_from_ds,
    make_hist_from_y,
    make_hist_anomaly_hampel,
    infer_freq_safe,
)

ID="unique_id"; DS="ds"; Y="y"

@dataclass
class FeatureConfig:
    freq: Optional[str]
    h: int
    calendar: Dict[str, Any]
    lag_features: Dict[str, Any]
    anomaly: Dict[str, Any]
    io: Dict[str, Any]
    parallel: Dict[str, Any]

def load_config(path: Optional[str]) -> FeatureConfig:
    import importlib.resources as ir
    if path is None:
        # load package default
        import pkgutil, json
        # fallback: open relative file
        with open(__file__.replace("runner_db.py","config_default.yaml"), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    return FeatureConfig(
        freq=cfg.get("freq"),
        h=int(cfg.get("h", 28)),
        calendar=cfg.get("calendar", {}),
        lag_features=cfg.get("lag_features", {}),
        anomaly=cfg.get("anomaly", {}),
        io=cfg.get("io", {}),
        parallel=cfg.get("parallel", {}),
    )

def _infer_series_freqs(df: pd.DataFrame) -> Dict[str,str]:
    freqs = {}
    for uid, g in df.groupby(ID, sort=False):
        freqs[uid] = infer_freq_safe(g[DS])
    return freqs

def _build_future_index(df: pd.DataFrame, h: int, freq_override: Optional[str]) -> pd.DataFrame:
    rows = []
    freqs = _infer_series_freqs(df) if freq_override is None else None
    for uid, g in df.groupby(ID, sort=False):
        last = pd.to_datetime(g[DS].max())
        freq = freq_override or freqs.get(uid, "D")
        try:
            future_ds = pd.date_range(last, periods=h+1, freq=freq, inclusive="neither")  # strictly after last
        except Exception:
            # fallback daily
            future_ds = pd.date_range(last, periods=h+1, freq="D", inclusive="neither")
        tmp = pd.DataFrame({ID: uid, DS: future_ds})
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[ID,DS])

def fetch_long(engine, table_fullname: str) -> pd.DataFrame:
    sql = text(f"SELECT * FROM {table_fullname} ORDER BY {DS}, num")
    return pd.read_sql(sql, engine)

def resolve_table_names(cfg_io: Dict[str,Any]) -> Tuple[str,str,str]:
    inp = cfg_io.get("table_input") or f"{TABLE_PREFIX}loto_final"
    hist = cfg_io.get("table_hist")  or f"{TABLE_PREFIX}features_hist"
    futr = cfg_io.get("table_futr")  or f"{TABLE_PREFIX}features_futr"
    return inp, hist, futr

def to_sql(df: pd.DataFrame, table: str, if_exists: str, engine, schema: Optional[str], chunksize: int=50000):
    df.to_sql(table, engine, schema=schema, index=False, if_exists=if_exists, chunksize=chunksize, method="multi")

def make_features_from_db(cfg_path: Optional[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(cfg_path)
    engine = get_engine()
    schema = cfg.io.get("schema")
    table_inp, table_hist, table_futr = resolve_table_names(cfg.io)

    # 1) fetch source
    df = fetch_long(engine, table_inp)
    # 必須列チェック
    for c in [ID, DS, Y]:
        if c not in df.columns:
            raise KeyError(f"入力テーブルに列 {c} がありません")
    df[DS] = pd.to_datetime(df[DS])
    df = df.sort_values([ID, DS]).reset_index(drop=True)

    # 2) build futr index
    futr_index = _build_future_index(df[[ID,DS]].drop_duplicates(), h=cfg.h, freq_override=cfg.freq)

    # 3) generate futr/hist features
    F_hist = make_futr_from_ds(df[[ID,DS]], cfg.calendar, ID, DS)
    F_futr = make_futr_from_ds(futr_index, cfg.calendar, ID, DS)

    H_base = make_hist_from_y(df[[ID,DS,Y]], cfg.lag_features, ID, DS, Y)

    # anomaly (hampel)
    anom_cfg = cfg.anomaly.get("hampel", {"windows":[28],"tau":3.5})
    H_anom = make_hist_anomaly_hampel(df[[ID,DS,Y]], anom_cfg, ID, DS, Y)

    # merge hist side
    X_hist = df[[ID,DS]].merge(H_base, on=[ID,DS], how="left").merge(H_anom, on=[ID,DS], how="left").merge(F_hist, on=[ID,DS], how="left")

    # merge futr side
    X_futr = futr_index.merge(F_futr, on=[ID,DS], how="left")

    # 4) write out
    write_mode = cfg.io.get("write_mode","replace")
    chunksize = int(cfg.io.get("chunksize", 50000))
    to_sql(X_hist, table_hist, if_exists=write_mode, engine=engine, schema=schema, chunksize=chunksize)
    to_sql(X_futr, table_futr, if_exists=write_mode, engine=engine, schema=schema, chunksize=chunksize)

    return X_hist, X_futr

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="DBから特徴量を生成し、DBに書き戻します（futr_/hist_ 接頭辞対応）。")
    p.add_argument("--config", type=str, default=None, help="YAML設定ファイルのパス（省略時は同梱のconfig_default.yaml）")
    args = p.parse_args()
    X_hist, X_futr = make_features_from_db(args.config)
    print(f"X_hist: {X_hist.shape}, X_futr: {X_futr.shape}")
