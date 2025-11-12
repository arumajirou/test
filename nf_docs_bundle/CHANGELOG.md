# 変更履歴（CHANGELOG）

## 2025-11-12
- `src/auto_model_factory.py` の構文エラー（トリプルクオートに混入したバックスラッシュ）を修正
- `create_auto_model()` を強化：DataFrame を受け取り内部で `TimeSeriesDataset` に変換、`predict()` の入力を統一
- 実行前バリデーション（ENVIRONMENT/BACKEND/MODEL/DATASET/HORIZON）を追加（`src/validation.py`）
- Optuna/Ray の切り替えロジックを整理（`src/search_algorithm_selector.py`）
- MLflow ロギングのオン/オフ切替に対応（`src/mlflow_integration.py`）
- ログ整形と例外可視化（`src/logging_config.py`）
- スモークスクリプトを追加（`smoke_optuna.py`, `smoke_ray.py`）
- Ray on Windows の既知問題を整理し、Linux（Kubuntu）実行を推奨
