# NeuralForecast AutoModels - 設定ファイル

NeuralForecast AutoModelsフレームワークの設定ファイルとヘルパーモジュールです。

## 📁 ファイル構成

```
outputs/
├── configs/
│   ├── model_characteristics.yaml   # モデル特性定義
│   └── default_configs.yaml         # デフォルト設定
├── config_loader.py                 # 設定読み込みヘルパー
├── example_config_usage.py          # 使用例
├── CONFIG_DESIGN.md                 # 設計ドキュメント
└── README.md                        # このファイル
```

## 🚀 クイックスタート

### 1. 基本的な使用方法

```python
from config_loader import load_default_config, get_model_config

# デフォルト設定を読み込む
config = load_default_config()

# MLflow設定にアクセス
print(config.mlflow.tracking_uri)  # "file:./mlruns"

# Optuna設定にアクセス
print(config.optuna.sampler.default)  # "TPESampler"

# モデル設定を取得
tft_config = get_model_config("AutoTFT")
print(tft_config.typical_search_space_size)  # 20000
```

### 2. MLflowとの統合

```python
from config_loader import load_default_config
from mlflow_integration import MLflowTracker

config = load_default_config()

# 設定を使ってMLflowTrackerを初期化
tracker = MLflowTracker(
    tracking_uri=config.mlflow.tracking_uri,
    experiment_name=config.mlflow.default_experiment_name
)

# 実験を開始
with tracker.start_run("trial_001"):
    tracker.log_params({"learning_rate": 0.001})
    tracker.log_metrics({"loss": 0.5})
```

### 3. 検索アルゴリズムの自動選択

```python
from config_loader import load_default_config, get_model_config
from search_algorithm_selector import SearchAlgorithmSelector

config = load_default_config()

# セレクターを初期化
selector = SearchAlgorithmSelector(
    backend=config.search_algorithm_selector.backend
)

# アルゴリズムを自動選択
strategy = selector.select_algorithm(
    model_complexity="complex",
    dataset_size=50000,
    num_samples=config.optuna.optimization.n_trials,
    config={"lr": 0.001, "batch_size": 32},
    use_pruning=config.optuna.pruner.enabled
)

print(f"選択されたアルゴリズム: {strategy.algorithm_name}")
```

## 📋 設定ファイル詳細

### model_characteristics.yaml

全AutoModelsの特性を定義しています：

- **モデル特性**：パラメータ種別、探索空間サイズ、ドロップアウト対応など
- **外生変数サポート**：各モデルがサポートする外生変数の種類
- **モデル複雑度分類**：Simple/Moderate/Complexの3段階

**主要なモデル**：
- Transformer系: AutoTFT, AutoAutoformer, AutoPatchTST, AutoInformer
- RNN系: AutoLSTM, AutoGRU, AutoTCN, AutoDeepAR
- NBEATS系: AutoNBEATS, AutoNBEATSx, AutoNHITS
- Linear系: AutoDLinear, AutoNLinear

### default_configs.yaml

フレームワーク全体のデフォルト設定を定義しています：

- **MLflow統合**: トラッキングサーバー、実験管理、自動ログ
- **Optuna最適化**: サンプラー、プルーナー、最適化パラメータ
- **Ray Tune**: 検索アルゴリズム、スケジューラー、チェックポイント
- **検索アルゴリズム選択**: 複雑度推定、推奨試行回数
- **ロギング**: ログレベル、出力先、フォーマット
- **検証**: クロスバリデーション、メトリクス、早期停止
- **学習**: エポック数、バッチサイズ、オプティマイザー

## 🔧 カスタマイズ

### 環境変数によるオーバーライド

環境変数を使って設定をオーバーライドできます：

```bash
# MLflow tracking URIを変更
export NEURALFORECAST_MLFLOW_TRACKING_URI="http://mlflow-server:5000"

# Optunaの試行回数を変更
export NEURALFORECAST_OPTUNA_OPTIMIZATION_N_TRIALS=100
```

```python
# 環境変数が自動的に適用される
config = load_default_config()
print(config.mlflow.tracking_uri)  # "http://mlflow-server:5000"
```

### 環境別設定ファイル

環境に応じた設定ファイルを作成できます：

```yaml
# configs/production.yaml
mlflow:
  tracking_uri: "databricks"
  default_experiment_name: "production_forecasting"

optuna:
  optimization:
    n_trials: 200
    n_jobs: 8

logging:
  level: "WARNING"
```

```python
# 環境を指定して読み込む
loader = ConfigLoader(env="production")
config = loader.load_config()
```

### プログラムからのオーバーライド

```python
# カスタム設定を用意
custom_config = {
    'mlflow': {
        'tracking_uri': 'http://custom-server:5000'
    },
    'training': {
        'max_epochs': 200
    }
}

# マージして読み込む
loader = ConfigLoader()
config = loader.load_config(override_config=custom_config)
```

## 📖 使用例

詳細な使用例は `example_config_usage.py` を参照してください：

```bash
python example_config_usage.py
```

実行すると以下の例が確認できます：

1. **基本的な読み込み**: 設定ファイルの基本的な読み込み方法
2. **モデル特性**: モデル特性の取得と活用
3. **MLflow統合**: MLflowTrackerでの使用方法
4. **検索アルゴリズム**: 自動選択機能の使用
5. **環境変数**: 環境変数によるオーバーライド
6. **カスタム設定**: カスタム設定のマージ
7. **複雑度分類**: モデル複雑度による分類
8. **設定保存**: カスタム設定の保存

## 📚 ドキュメント

詳細な設計思想と技術仕様は `CONFIG_DESIGN.md` を参照してください：

- 設計思想とアーキテクチャ
- 各設定項目の詳細説明
- ベストプラクティス
- トラブルシューティング
- 今後の拡張予定

## 🔍 主要APIリファレンス

### ConfigLoader

```python
class ConfigLoader:
    """設定ファイル読み込みクラス"""
    
    def __init__(self, config_dir: str = "configs", env: str = None)
    def load_config(self, config_name: str = "default_configs.yaml") -> ConfigDict
    def load_model_characteristics(self, config_name: str = "model_characteristics.yaml") -> ConfigDict
    def get_model_config(self, model_name: str) -> ConfigDict
    def save_config(self, config: Dict, output_path: str) -> None
```

### 便利関数

```python
# デフォルト設定を読み込む
config = load_default_config(config_dir="configs", env=None)

# モデル特性を読み込む
model_chars = load_model_characteristics(config_dir="configs")

# 特定のモデル設定を取得
model_config = get_model_config(model_name="AutoTFT", config_dir="configs")
```

## 💡 推奨されるワークフロー

### 1. 開発環境での使用

```python
# デフォルト設定で開発
config = load_default_config()

# 必要に応じて一部オーバーライド
config.training.max_epochs = 10  # 開発時は短く
config.optuna.optimization.n_trials = 5
```

### 2. 本番環境への展開

```bash
# 環境変数を設定
export ENV="production"
export NEURALFORECAST_MLFLOW_TRACKING_URI="databricks"
```

```python
# 本番環境設定が自動適用される
config = load_default_config()
```

### 3. 実験管理

```python
# 実験ごとに設定を保存
loader = ConfigLoader()
config = loader.load_config()

# 実験に使った設定を保存
experiment_name = "tft_experiment_001"
loader.save_config(config, f"experiments/{experiment_name}/config.yaml")
```

## ⚙️ システム要件

- Python 3.8+
- PyYAML
- その他の依存関係は requirements.txt を参照

## 🤝 貢献

設定項目の追加や変更の提案は、以下の手順で行ってください：

1. `model_characteristics.yaml` または `default_configs.yaml` を編集
2. `CONFIG_DESIGN.md` にドキュメントを追加
3. `example_config_usage.py` に使用例を追加
4. 変更をテスト

## 📝 ライセンス

このプロジェクトのライセンスについては、メインプロジェクトのライセンスを参照してください。

## 📞 サポート

質問や問題が発生した場合：

1. `CONFIG_DESIGN.md` のトラブルシューティングセクションを確認
2. `example_config_usage.py` で類似の使用例を探す
3. ログを確認（`logs/neuralforecast_auto.log`）

---

**最終更新**: 2025-01-15
**バージョン**: 1.0.0
