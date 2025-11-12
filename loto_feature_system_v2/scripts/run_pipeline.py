#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from src.pipelines.loto_pipeline import run as run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run LOTO feature engineering pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV (unique_id, ds, y)")
    parser.add_argument("--output-dir", required=True, help="Directory to save features")
    parser.add_argument("--config", required=True, help="Pipeline config YAML")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = run_pipeline(args.input, args.output_dir, args.config)
    for k, p in results.items():
        print(f"{k}: {p}")

if __name__ == "__main__":
    main()
