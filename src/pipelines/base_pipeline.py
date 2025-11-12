from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class PipelineIO:
    input_csv: str
    output_dir: str
    config_path: str

def load_pipeline_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
