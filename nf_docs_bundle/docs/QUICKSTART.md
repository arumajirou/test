# クイックスタート

本書は最短で動作確認するための手順を記載します。詳細はルートの `README.md` を参照してください。

## 1) 環境作成（Kubuntu 推奨）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

## 2) PyTorch のインストール

CUDA の有無に応じて公式手順に従ってください。例（CPU のみ）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3) 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 4) データの準備

`N4.csv` のような日付列+数値列からでも、自動で `unique_id, ds, y` へ変換します。
形式の詳細は `DATA_FORMAT.md` 参照。

## 5) スモーク実行

### Optuna（1 trial）
```bash
export PYTHONPATH=$(pwd)
export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
export MLFLOW_EXPERIMENT_NAME="neuralforecast_demo"
python smoke_optuna.py
```

### Ray（1 trial, Linux 推奨）
```bash
export PYTHONPATH=$(pwd)
python smoke_ray.py
```

## 6) 成果物の場所
- `predictions_*.csv` … 予測出力
- `mlruns/` … MLflow ログ
- `nf_auto_runs/<run_id>/` … 学習チェックポイント（`models/`）、メトリクス/予測テーブル（`tables/`）
