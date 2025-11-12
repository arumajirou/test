from pathlib import Path
from src.core.data_loader import load_from_csv

def test_load_from_csv(tmp_path: Path):
    p = tmp_path / "mini.csv"
    p.write_text("unique_id,ds,y\na,2024-01-01,1\n")
    df = load_from_csv(p)
    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert len(df) == 1
