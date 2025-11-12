#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_patches_auto_model_factory.py
- C:\test\src\auto_model_factory.py に対して以下の修正を自動適用します。
  1) MLflow 親 Run 検出→ nested=True で開始
  2) Auto* クラス生成時に未知キーワード（例: n_series）を自動フィルタ
  3) active_experiment_id() 非依存化（既に置換済みでも再実行安全）
"""
import re
import sys
from pathlib import Path
from datetime import datetime

TARGET = Path(r"C:\test\src\auto_model_factory.py")

def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return bak

def ensure_helper_function(src: str) -> (str, bool):
    """_instantiate_model ヘルパーを先頭のimport群の後へ挿入"""
    if "_instantiate_model(" in src:
        return src, False
    helper = r