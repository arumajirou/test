# 自動モデルファクトリー - NeuralForecast自動最適化システム

## 📋 概要

このプロジェクトは、NeuralForecastの時系列予測モデルに対して、Optuna/Ray Tuneを使用した自動ハイパーパラメータ最適化を提供する統合システムです。

### 主な特徴

- ✅ **包括的な検証**: 設定、環境、データの自動検証
- 🎯 **最適なアルゴリズム選択**: モデル・データ特性に基づく自動選択
- 📊 **MLflow統合**: 実験追跡と管理
- 🔧 **柔軟な設定**: デフォルトとカスタム設定の両対応
- 🚀 **簡単な使用**: 数行のコードで実行可能

## 📁 ファイル構成

```
.
├── validation.py                    # 設定検証モジュール
├── search_algorithm_selector.py     # 探索アルゴリズム選択
├── auto_model_factory.py            # メインファクトリー
├── DESIGN_OVERVIEW.md               # 詳細な設計ドキュメント
├── example_usage.py                 # 使用例スクリプト
└── README.md                        # このファイル
```

## 🚀 クイックスタート

### 1. 基本的な使用方法

```python
import pandas as pd
from auto_model_factory import create_auto_model

# データ準備（必須カラム: unique_id, ds, y）
df = pd.read_csv('your_timeseries.csv')

# 自動最適化実行
auto_model = create_auto_model(
    model_name="NHITS",      # モデル選択
    h=24,                     # 予測ホライゾン
    dataset=df,               # データ
    backend="optuna",         # optunaまたはray
    num_samples=50,           # 試行回数
    use_mlflow=True,          # 実験追跡
    verbose=True              # 詳細出力
)

# 予測実行
predictions = auto_model.predict(dataset=df)
```

### 2. カスタム設定での使用

```python
from ray import tune
from auto_model_factory import create_auto_model

# ハイパーパラメータ探索空間の定義
custom_config = {
    'max_steps': tune.choice([1000, 2000, 3000]),
    'learning_rate': tune.loguniform(1e-4, 1e-2),
    'batch_size': tune.choice([64, 128, 256]),
    'input_size': tune.choice([14, 28, 56])
}

# カスタム設定で最適化
auto_model = create_auto_model(
    model_name="TFT",
    h=24,
    dataset=df,
    config=custom_config,
    num_samples=100,
    cpus=8,
    gpus=2
)
```

### 3. ファクトリークラスでの高度な使用

```python
from auto_model_factory import AutoModelFactory, OptimizationConfig

# 最適化設定
opt_config = OptimizationConfig(
    backend="optuna",
    num_samples=None,        # 自動推奨
    cpus=8,
    gpus=2,
    use_mlflow=True,
    mlflow_experiment_name="production_forecast",
    use_pruning=True,
    time_budget_hours=3.0,
    random_seed=42
)

# ファクトリー作成
factory = AutoModelFactory(
    model_name="TFT",
    h=24,
    optimization_config=opt_config
)

# 最適化実行
auto_model = factory.create_auto_model(dataset=df)

# 最適化履歴の確認
history = factory.get_optimization_summary()
print(history)
```

## 📚 各モジュールの詳細

### validation.py - 設定検証モジュール

**機能**:
- バックエンド設定の検証（Optuna/Ray）
- モデル設定の検証
- 実行環境の検証（GPU、メモリ、ディスク）
- データセット検証
- MLflow設定検証

**使用例**:
```python
from validation import validate_all, print_validation_results

results = validate_all(
    backend="optuna",
    config=my_config,
    num_samples=50,
    cpus=4,
    gpus=1,
    model_class_name="NHITS",
    dataset=df,
    h=24
)

print_validation_results(results)
```

### search_algorithm_selector.py - 探索アルゴリズム選択

**機能**:
- モデル複雑度とデータサイズに基づくアルゴリズム選択
- Optunaサンプラー/プルーナーの自動設定
- Ray Tuneサーチアルゴリズムの自動選択
- 試行回数の推奨

**使用例**:
```python
from search_algorithm_selector import (
    SearchAlgorithmSelector,
    ModelComplexity,
    DatasetSize,
    recommend_num_samples
)

# アルゴリズム選択
selector = SearchAlgorithmSelector(backend="optuna")
strategy = selector.select_algorithm(
    model_complexity=ModelComplexity.COMPLEX,
    dataset_size=DatasetSize.LARGE,
    num_samples=100,
    config=my_config
)

# 試行回数推奨
num_samples, explanation = recommend_num_samples(
    model_complexity=ModelComplexity.MODERATE,
    dataset_size=DatasetSize.MEDIUM,
    search_complexity=SearchComplexity.HIGH,
    time_budget_hours=2.0
)
```

### auto_model_factory.py - メインファクトリー

**機能**:
- 全コンポーネントの統合
- 5段階の最適化フロー
- MLflow統合
- モデルカタログ管理
- 最適化履歴の記録

**最適化フロー**:
1. 検証 (Validation)
2. データセット分析 (Dataset Analysis)
3. 探索戦略選択 (Search Strategy Selection)
4. ハイパーパラメータ設定 (Configuration)
5. モデル作成と最適化 (Optimization)

## 🎯 サポートされるモデル

| モデル | 複雑度 | 推奨input_size | 典型的な学習時間 |
|--------|--------|----------------|------------------|
| MLP | Simple | 7-14 | 5分 |
| NHITS | Moderate | 14-28 | 10分 |
| NBEATS | Moderate | 14-28 | 10分 |
| DLinear | Moderate | 24-96 | 8分 |
| TSMixer | Moderate | 24-96 | 15分 |
| TFT | Complex | 24-168 | 30分 |
| Transformer | Complex | 24-96 | 25分 |
| PatchTST | Complex | 96-512 | 20分 |

## 📊 使用例スクリプト

`example_usage.py`には6つの実用的な例が含まれています:

1. **基本的な使用方法** - シンプルな自動最適化
2. **検証機能の使用** - 包括的な検証の実行
3. **探索アルゴリズムの選択** - アルゴリズム選択のデモ
4. **高度なファクトリーの使用** - ファクトリークラスの活用
5. **カスタム設定の使用** - カスタムハイパーパラメータ空間
6. **データ検証の詳細** - データ検証機能のデモ

実行方法:
```bash
python example_usage.py
```

## 🔧 インストール要件

```bash
# 必須パッケージ
pip install neuralforecast
pip install optuna
pip install 'ray[tune]'
pip install mlflow
pip install pytorch-lightning
pip install pandas numpy

# GPU使用の場合
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 💡 ベストプラクティス

### 1. 段階的アプローチ

```python
# まず少ない試行でテスト
quick_test = create_auto_model(..., num_samples=5)

# 良好な結果なら本格的な最適化
full_run = create_auto_model(..., num_samples=100)
```

### 2. リソース管理

```python
# 大規模データの場合
opt_config = OptimizationConfig(
    num_samples=30,          # 試行回数を抑える
    cpus=16,
    gpus=4,                  # 複数GPU
    use_pruning=True         # 早期停止
)

# 小規模データの場合
opt_config = OptimizationConfig(
    num_samples=100,         # 多めの試行
    use_pruning=False,       # プルーニング無効
    time_budget_hours=None   # 時間制限なし
)
```

### 3. MLflowでの追跡

```python
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run(run_name="forecast_v1"):
    auto_model = create_auto_model(...)
    
    # カスタムメトリクスの追加
    mlflow.log_metrics({
        'dataset_size': len(df),
        'n_series': df['unique_id'].nunique()
    })
```

## 🐛 トラブルシューティング

### OOMエラー

```python
# 解決策
config = {
    'batch_size': tune.choice([16, 32]),  # 小さいバッチサイズ
    'input_size': tune.choice([7, 14])     # 短いルックバック
}
opt_config = OptimizationConfig(gpus=2)    # GPU増やす
```

### 収束しない

```python
# 試行回数を増やす
opt_config = OptimizationConfig(num_samples=200)

# 探索空間を狭める
config = {
    'learning_rate': tune.loguniform(1e-4, 1e-3)  # 狭い範囲
}
```

### MLflow接続エラー

```python
from validation import ConfigValidator

validator = ConfigValidator()
result = validator.validate_mlflow_config(
    tracking_uri="http://localhost:5000"
)
print(result.errors)
```

## 📖 詳細ドキュメント

より詳細な情報は`DESIGN_OVERVIEW.md`を参照してください:

- アーキテクチャの詳細
- 各モジュールの設計思想
- アルゴリズム選択ロジック
- 高度な使用例
- パフォーマンスチューニング

## 🤝 貢献

このプロジェクトへの貢献を歓迎します。以下の方法で貢献できます:

1. バグ報告
2. 機能リクエスト
3. コード改善
4. ドキュメント改善

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🔗 関連リンク

- [NeuralForecast Documentation](https://nixtla.github.io/neuralforecast/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## 📧 サポート

問題や質問がある場合は、GitHubのIssuesを使用してください。

---

**作成日**: 2025年11月12日  
**バージョン**: 1.0.0
