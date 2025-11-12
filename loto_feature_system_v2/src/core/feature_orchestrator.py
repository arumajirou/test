from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
import numpy as np

from ..utils.logging_config import get_logger
from ..utils.validation import ensure_columns

logger = get_logger(__name__)

@dataclass
class FeatureConfig:
    lags: list[int]
    roll_mean: list[int]
    roll_std: list[int]
    diffs: list[int]
    futr_calendar_fields: list[str]
    stat_fields: list[str]

class FeatureOrchestrator:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame, cfg: FeatureConfig) -> dict[str, Path]:
        ensure_columns(df)
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # ---------- HISTORICAL FEATURES ----------
        hist = df[["unique_id", "ds", "y"]].copy()
        for lag in cfg.lags:
            hist[f"hist_lag_{lag}"] = hist.groupby("unique_id")["y"].shift(lag)
        for w in cfg.roll_mean:
            hist[f"hist_rollmean_{w}"] = (
                hist.groupby("unique_id")["y"].transform(lambda s: s.rolling(w, min_periods=max(1, w//2)).mean())
            )
        for w in cfg.roll_std:
            hist[f"hist_rollstd_{w}"] = (
                hist.groupby("unique_id")["y"].transform(lambda s: s.rolling(w, min_periods=max(1, w//2)).std())
            )
        for d in cfg.diffs:
            hist[f"hist_diff_{d}"] = hist.groupby("unique_id")["y"].diff(d)

        # ---------- FUTURE-KNOWN (CALENDAR) FEATURES ----------
        futr = df[["unique_id", "ds"]].copy()
        futr["year"] = futr["ds"].dt.year
        futr["month"] = futr["ds"].dt.month
        futr["day"] = futr["ds"].dt.day
        futr["weekday"] = futr["ds"].dt.weekday
        futr["week"] = futr["ds"].dt.isocalendar().week.astype(int)
        futr["quarter"] = futr["ds"].dt.quarter
        futr["is_month_start"] = futr["ds"].dt.is_month_start.astype(int)
        futr["is_month_end"] = futr["ds"].dt.is_month_end.astype(int)

        # keep only requested fields
        keep = ["unique_id", "ds"] + [c for c in cfg.futr_calendar_fields if c in futr.columns]
        futr = futr[keep]

        # ---------- STATIC FEATURES ----------
        stat = df[["unique_id"]].drop_duplicates().copy()
        if "unique_id_len" in cfg.stat_fields:
            stat["unique_id_len"] = stat["unique_id"].astype(str).str.len()

        # ---------- SAVE ----------
        hist_path = self.output_dir / "features_hist.csv"
        futr_path = self.output_dir / "features_futr.csv"
        stat_path = self.output_dir / "features_stat.csv"
        hist.to_csv(hist_path, index=False)
        futr.to_csv(futr_path, index=False)
        stat.to_csv(stat_path, index=False)

        logger.info("Saved features: %s, %s, %s", hist_path, futr_path, stat_path)
        return { "hist": hist_path, "futr": futr_path, "stat": stat_path }
