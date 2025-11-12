from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..core.data_loader import load_from_csv
from ..core.feature_orchestrator import FeatureOrchestrator, FeatureConfig
from .base_pipeline import load_pipeline_config

def run(input_csv: str, output_dir: str, config_path: str) -> dict[str, Path]:
    cfg_yaml = load_pipeline_config(config_path)
    df = load_from_csv(input_csv)

    fc = FeatureConfig(
        lags=cfg_yaml["features"]["hist"]["lags"],
        roll_mean=cfg_yaml["features"]["hist"]["roll_mean"],
        roll_std=cfg_yaml["features"]["hist"]["roll_std"],
        diffs=cfg_yaml["features"]["hist"]["diffs"],
        futr_calendar_fields=cfg_yaml["features"]["futr_calendar"]["fields"],
        stat_fields=cfg_yaml["features"]["stat"]["fields"],
    )

    orch = FeatureOrchestrator(output_dir)
    return orch.run(df, fc)
