# NeuralForecast Auto Model Factory（拡張版）

本リポジトリは **NeuralForecast** を用いた時系列 Auto*（AutoNHITS/AutoTFT/AutoDLinear など）を
簡潔に扱うための工場モジュールです。**Optuna** と **Ray Tune** の両バックエンドに対応し、
**MLflow** による実験管理をオン/オフ切替できます。

> **重要**: Ray は Windows 環境でダッシュボード/エージェントの不具合により不安定になる例が確認されています。
> **Kubuntu（Linux）での実行を推奨**します。詳細は `docs/RAY_ON_WINDOWS.md` を参照してください。

---

## 特長

- `create_auto_model(model_name, h, dataset, backend, ...)` の単一エントリポイント
- `pandas.DataFrame`（`unique_id, ds, y`）をそのまま受け取り、内部で `TimeSeriesDataset` に変換
- `backend="optuna" | "ray"` を統一引数で切替
- `use_mlflow=True/False` で MLflow ロギングを制御
- 実行前バリデーション（ENVIRONMENT / BACKEND / MODEL / DATASET / HORIZON）

---

## リポジトリ構成（抜粋）

```
.
├─ src/
│  ├─ auto_model_factory.py        # 工場メソッドの実装
│  ├─ validation.py                # 実行時バリデーション
│  ├─ search_algorithm_selector.py # Optuna/Ray の選択ロジック
│  ├─ mlflow_integration.py        # MLflow 連携
│  ├─ logging_config.py            # ログ設定
│  └─ __init__.py
├─ configs/                        # 設定・モデル特性
├─ docs/
│  ├─ QUICKSTART.md
│  ├─ DATA_FORMAT.md
│  └─ RAY_ON_WINDOWS.md
├─ smoke_optuna.py                 # 最小疎通（Optuna, 1 trial）
├─ smoke_ray.py                    # 最小疎通（Ray, 1 trial）
└─ requirements.txt
```

---

## 要件

- Python 3.11 推奨（3.10 以上を想定）
- **PyTorch**（CUDA の有無は環境に合わせて公式手順でインストール）
- 以降の Python 依存は `requirements.txt` を参照  
  ※ Ray は Linux を推奨。Windows ではインストール自体は可能でも動作が不安定な事例があります。

---

## インストール

### 1) 仮想環境作成（例：Kubuntu）
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) PyTorch（公式手順で）
```bash
# 例: CPU のみ
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3) 依存関係
```bash
pip install -U pip
pip install -r requirements.txt
```

---

## データ形式

NeuralForecast の標準形式 **`unique_id, ds, y`** を採用します。
詳細と自動変換ロジックの注意点は `docs/DATA_FORMAT.md` を参照してください。

---

## クイックスタート

### A. Optuna（1 trial のスモーク）
```bash
export PYTHONPATH=$(pwd)
export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
export MLFLOW_EXPERIMENT_NAME="neuralforecast_demo"

python smoke_optuna.py
```

出力例：
- 予測CSV：`predictions_N4_optuna.csv`
- MLflow：`mlruns/` 配下にパラメータ・メトリクス・タグが保存
- NF 実行成果：`nf_auto_runs/<run_id>/` 配下に `models/` と `tables/`

### B. Ray（1 trial のスモーク; Linux 推奨）
```bash
export PYTHONPATH=$(pwd)
python smoke_ray.py
```

> Windows での Ray 実行に関する既知の問題は `docs/RAY_ON_WINDOWS.md` を参照。

---

## MLflow の使い方（任意）

- ローカル記録：`MLFLOW_TRACKING_URI="file://<abs path>/mlruns"`
- 既定の実験名：環境変数 `MLFLOW_EXPERIMENT_NAME` で指定
- `use_mlflow=True` にするとパラメータ/メトリクス/タグ/アーティファクトが自動記録されます。

---

## トラブルシューティング

- CUDA が無効: CPU 実行にフォールバックします（警告表示）。
- メモリ不足: `batch_size` を下げる、`max_steps` を段階的に上げる。
- Optuna の categorical 警告（タプル混入）: 学習は継続可能。探索空間をプリミティブ列挙に整理してください。
- Ray on Windows: ダッシュボード/エージェントの不具合が既知。Linux での実行を推奨。

---

## ライセンス

プロジェクトのライセンス方針に従ってください（未定の場合は社内標準に合わせてください）。

---

## 変更履歴

`CHANGELOG.md` を参照（2025-11-12 版）。
