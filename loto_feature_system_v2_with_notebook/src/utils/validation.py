from typing import Sequence
import pandas as pd

REQUIRED_COLS = ["unique_id", "ds", "y"]

def ensure_columns(df: pd.DataFrame, required: Sequence[str] = REQUIRED_COLS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
