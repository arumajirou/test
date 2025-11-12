from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..utils.validation import ensure_columns

def load_from_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    ensure_columns(df)
    # coerce ds as datetime if possible
    try:
        df["ds"] = pd.to_datetime(df["ds"])
    except Exception:
        pass
    return df
