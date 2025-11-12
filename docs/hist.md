# 修正・改修履歴報告書（NeuralForecast Auto Model Factory 一式）

* 対象パス：`C:\test`（Windows 検証環境）
* 今後の実行環境：Kubuntu（Linux）へ移行
* 作業日：2025-11-12（JST）

---

## 1. 目的

* `src/auto_model_factory.py` の構文エラー解消と工場メソッドの安定化。
* Optuna/Ray バックエンドの最小疎通と、MLflow 連携のオン/オフ切替。
* N4.csv を用いたスモーク（予測CSV保存）とログ出力確認。
* Windows で発生した Ray のダッシュボード起因クラッシュを切り分け、Linux へ移行判断。

---

## 2. 変更サマリ（概要）

1. **構文エラー修正**

   * `auto_model_factory.py` 417行付近のドキュメンテーション文字列で、`\"\"\"...\"\"\"` のような **バックスラッシュ付きクオート**が残存し、`SyntaxError: unexpected character after line continuation character` を誘発。
   * 余計なバックスラッシュを除去し、**正規のトリプルクオート `"""..."""`** に統一。その他、隣接する行の空白・改行を整理。

2. **工場メソッドの堅牢化**

   * `create_auto_model()` が **`pandas.DataFrame` をそのまま受け取り**、内部で `TimeSeriesDataset` へ変換する導線を整備。
   * `predict(dataset=...)` が **DataFrame / TimeSeriesDataset のどちらでも動く**ように分岐とバリデーションを実装。
   * バックエンド：`backend="optuna" | "ray"` の切替、および `num_samples`・`config`・`use_mlflow`・`verbose` を統一引数で受付。
   * 例外時のメッセージ整備（モデル未実装・データ形式不正・ハイパラ定義漏れの検出）。

3. **バリデーション機構の追加**（`src/validation.py`）

   * 起動時に **ENVIRONMENT / BACKEND / MODEL / DATASET / HORIZON** をチェックし、`✓ PASSED` と警告を出力。
   * 検出した主な警告：

     * CUDA 無効（CPU実行）
     * 低メモリ（1.13 GB）
     * 一部の推奨ハイパラ未指定（`learning_rate` / `batch_size` / `max_steps` など）

4. **Optuna 連携**（`search_algorithm_selector.py` 経由）

   * `num_samples=1` のスモークを安定実行。
   * Categorical に **タプルが混入している場合の Optuna 警告**は、学習自体は継続できるため情報レベルで許容（値集合が `(1,1,1)` 等の冗長タプルになっていた箇所は、順次 **プリミティブな列挙**に置き換える方針）。

5. **MLflow 連携**（`src/mlflow_integration.py`）

   * `use_mlflow=True/False` で制御。ローカルファイル URI（例：`file:///C:/test/mlruns`）に対応。
   * 実行パラメータ・メトリクスの軽量ロギング、タグ（`backend` など）付与。
   * 学習結果は `mlruns/929142664854178861/...` などに格納（パラメータ・メトリクス・タグ配下を確認済み）。

6. **ログ設定**（`src/logging_config.py`）

   * コンソール出力の整形、冗長ログの抑制、例外の見える化。

7. **スモークスクリプト配置**

   * ルート直下に `smoke_optuna.py` / `smoke_ray.py` を配置（最小疎通用、N4.csv 対応）。

---

## 3. 変更ファイル一覧（主要）

* `src/auto_model_factory.py` … 構文修正・データ受け渡し・predict 導線統一・例外処理
* `src/validation.py` … バリデーション一式（5カテゴリ）
* `src/search_algorithm_selector.py` … Optuna/Ray の切替実装
* `src/mlflow_integration.py` … MLflow ログ連携（オン/オフ）
* `src/logging_config.py` … ロギング定義
* バックアップ：`src/auto_model_factory.py.bak_YYYYmmdd_HHMMSS` が多数生成（復旧用スナップショット）

---

## 4. テスト・検証結果（抜粋）

### 4.1 構文チェック

* コマンド：`python -m py_compile C:\test\src\auto_model_factory.py`
* 結果：**無エラーで完了**（修正前の 417行付近エラーは再現せず）

### 4.2 Optuna スモーク（N4.csv／1 trial）

* 実行：PowerShell ワンライナーおよび `smoke_optuna.py` 同等の処理
* ログ抜粋：

  * `Validation: ... ✓ PASSED`（5カテゴリすべて合格）
  * 警告（情報）：CUDA 無効、Low Memory、推奨ハイパラ未指定
* 生成物：

  * MLflow：`mlruns/929142664854178861/...` 配下に **params/metrics/tags** が保存
  * NF ラン：`nf_auto_runs/run_20251112-075246/` に **models/**（ckpt 等）と **tables/**（`cv_predictions.csv`, `metrics.csv`）
  * 予測CSV：都度のスクリプトで `predictions_N4_optuna.csv` を出力可能（実行時の引数に依存）

### 4.3 Ray スモーク（Windows）

* 事象：

  * `Failed to start the dashboard`, `ValueError: Must have exactly one of read or write mode`
  * `dashboard_agent failed and raylet fate-shares with it` により **raylet が即時終了**
  * `Unable to register worker with raylet` で Tune 実行不可
* 切り分け：

  * ダッシュボード抑止・ログリダイレクト抑止・`local_mode=True`・`address="local"` 等の組合せを試行するも、Windows では安定せず。
* **結論**：**Kubuntu（Linux）へ移行して Ray を実行**する方針を採用。

---

## 5. 既知の課題・注意事項

* **Ray on Windows**：ダッシュボード／エージェントのロギング初期化で例外が発生し、raylet が運命共同体(fate-share)で終了。Windows では再発可能性が高く、Linux 移行が合理的。
* **メモリ**：ローカル環境の空きが少なく、学習バッチは保守的に（`batch_size` を小さめに、`max_steps` も段階的に）。
* **Optuna 警告**：categorical にタプルが含まれる警告は学習継続可能だが、**順次プリミティブ列挙へ整理**するのが望ましい。

---

## 6. Kubuntu 実行ガイド（推奨）

* Python 3.11 系、`neuralforecast`, `optuna`, `mlflow`, `ray[tune]` をインストール。
* Ray は **安定版（例：2.31系）**から開始し、Tune の疎通 → NF 学習の順に確認。
* 実行テンプレ（概略）：

  1. `ray.init(local_mode=True, include_dashboard=False, num_cpus=<物理に合わせる>)`
  2. N4.csv 整形（`unique_id, ds, y` へ）
  3. `create_auto_model("DLinear" or "NHITS", backend="ray", num_samples=1, ...)`
  4. `predict()` → `predictions_*.csv` 保存
  5. `ray.shutdown()`

---

## 7. 運用上の推奨

* **最小スモーク（1 trial）→ 本番試行**の二段階で回す。まず DLinear で疎通、その後 NHITS/TFT へ拡張。
* MLflow はローカル `file://` で十分にトラッキング可能。Kubuntu ではパス権限のみ注意。
* 学習が重いモデルは **`max_steps` 漸増・`batch_size` スケール**で OOM を防止。

---

## 8. 付録：再現コマンド（要約）

### 8.1 構文チェック

```powershell
python -m py_compile C:\test\src\auto_model_factory.py
```

### 8.2 Optuna（N4.csv／1 trial／予測保存）

```powershell
$env:PYTHONPATH="C:\test"
$env:MLFLOW_TRACKING_URI="file:///C:/test/mlruns"
$env:MLFLOW_EXPERIMENT_NAME="neuralforecast_demo"
python C:\test\smoke_optuna.py
```

### 8.3 Ray（Linux 想定の最小例）

```bash
python - << 'PY'
import os, ray
os.environ["RAY_DISABLE_DASHBOARD"]="1"
ray.init(local_mode=True, include_dashboard=False, num_cpus=1)
print(ray.cluster_resources())
ray.shutdown()
PY
```

その後、Optuna と同等の整形→`create_auto_model(..., backend="ray")`→`predict()` を実行。

---

## 9. 成果物・生成物（確認済み）

* **MLflow**：`mlruns/929142664854178861` 配下の各ラン（params/metrics/tags/artifacts）
* **NF 実行成果**：`nf_auto_runs/run_20251112-075246/`

  * `models/`（`AutoPatchTST_0.ckpt`, `AutoTFT_0.ckpt`, ほか）
  * `tables/`（`cv_predictions.csv`, `metrics.csv`）
* **スモークスクリプト**：`smoke_optuna.py`, `smoke_ray.py`
* **バックアップ**：`src/auto_model_factory.py.bak_*`（差分復旧用に保持）

---

## 10. 今後の改善候補

* Optuna の **探索空間からタプル集合を排除**（警告根絶）
* モデル別の **推奨デフォルトハイパラ表**を `configs/` に明示化
* Kubuntu での **Ray バージョン固定・E2E テンプレ**の整備（再現性担保）
* 低メモリ環境向けの **省メモリプリセット**（`batch_size`, `num_workers`, `input_size`）の追加

---

以上。Kubuntu での Ray 再検証に合わせて、探索空間の整理と学習プリセットを拡充する。
