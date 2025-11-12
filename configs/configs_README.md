# Configuration Files Guide

このディレクトリには、NeuralForecast自動最適化システムの設定ファイルが含まれています。

## ファイル構成

```
configs/
├── model_characteristics.yaml   # モデル特性定義（28モデル）
├── default_configs.yaml         # システムデフォルト設定
└── README.md                    # このファイル
```

---

## 1. model_characteristics.yaml

### 概要
全28種類のNeuralForecast AutoModelsの特性を定義したファイルです。各モデルの最適化タイプ、パラメータ特性、推奨設定などを一元管理します。

### データ構造

各モデルは以下の構造で定義されています：

```yaml
AutoTFT:
  model_name: "AutoTFT"                    # モデル名
  base_model_name: "TFT"                   # ベースモデル名
  model_category: "transformer"            # カテゴリ
  complexity: "complex"                    # 複雑度
  
  parameters:                              # パラメータ特性
    has_continuous: true                   # 連続パラメータの有無
    has_discrete: true                     # 離散パラメータの有無
    has_categorical: true                  # カテゴリカルパラメータの有無
    typical_search_space_size: 20000       # 探索空間サイズ
  
  optimization:                            # 最適化設定
    type: "mixed"                          # 最適化タイプ
    supports_multi_objective: false        # 多目的最適化対応
    dropout:                               # ドロップアウト設定
      supported: true
      key: "dropout"
      default: 0.1
    early_stop_default: -1                 # 早期停止デフォルト
  
  scaler:                                  # スケーラー設定
    default: "identity"
    choices: ["identity", "standard", "robust", "minmax"]
  
  exogenous:                              # 外生変数サポート
    support: "full"                       # none/futr_only/hist_only/stat_only/full
  
  backend:                                # バックエンド設定
    choices: ["ray", "optuna"]
    pruner_default_enabled_optuna: true
  
  notes: "Temporal Fusion Transformer, excellent with exogenous variables"
```

### モデルカテゴリ

| カテゴリ | モデル数 | 例 |
|---------|---------|-----|
| **transformer** | 7 | AutoTFT, AutoAutoformer, AutoPatchTST |
| **rnn** | 6 | AutoLSTM, AutoGRU, AutoTCN |
| **linear** | 4 | AutoDLinear, AutoNLinear, AutoMLP |
| **nbeats** | 3 | AutoNBEATS, AutoNBEATSx, AutoNHITS |
| **other** | 8 | AutoTiDE, AutoTimeMixer, AutoStemGNN |

### 複雑度分類

| 複雑度 | 説明 | 推奨試行回数 |
|--------|------|-------------|
| **simple** | 軽量モデル | 20-30回 |
| **moderate** | 中規模モデル | 50-100回 |
| **complex** | 大規模モデル | 100-200回 |

### 使用例

#### Python内でYAMLを読み込む

```python
import yaml

# YAMLファイルを読み込む
with open('configs/model_characteristics.yaml', 'r') as f:
    model_chars = yaml.safe_load(f)

# 特定のモデルの特性を取得
tft_char = model_chars['AutoTFT']
print(f"Model: {tft_char['model_name']}")
print(f"Complexity: {tft_char['complexity']}")
print(f"Dropout default: {tft_char['optimization']['dropout']['default']}")
print(f"Exogenous support: {tft_char['exogenous']['support']}")

# カテゴリ別にモデルをフィルタリング
transformer_models = [
    name for name, char in model_chars.items()
    if char['model_category'] == 'transformer'
]
print(f"Transformer models: {transformer_models}")

# ドロップアウト対応モデルを検索
dropout_models = [
    name for name, char in model_chars.items()
    if char['optimization']['dropout']['supported']
]
print(f"Models with dropout: {dropout_models}")
```

#### モデル選択のヘルパー関数

```python
def select_model_by_requirements(
    model_chars: dict,
    exogenous_needed: bool = False,
    max_complexity: str = 'complex',
    min_search_space: int = 0
) -> list:
    """要件に基づいてモデルを選択"""
    complexity_order = {'simple': 0, 'moderate': 1, 'complex': 2}
    max_comp_level = complexity_order[max_complexity]
    
    suitable_models = []
    for name, char in model_chars.items():
        # 複雑度チェック
        if complexity_order[char['complexity']] > max_comp_level:
            continue
        
        # 外生変数チェック
        if exogenous_needed and char['exogenous']['support'] == 'none':
            continue
        
        # 探索空間サイズチェック
        if char['parameters']['typical_search_space_size'] < min_search_space:
            continue
        
        suitable_models.append(name)
    
    return suitable_models

# 使用例
models = select_model_by_requirements(
    model_chars,
    exogenous_needed=True,
    max_complexity='moderate'
)
print(f"Suitable models: {models}")
```

---

## 2. default_configs.yaml

### 概要
システム全体のデフォルト設定を定義したファイルです。実験追跡、ハイパーパラメータ最適化、分散計算など、12のセクションで構成されています。

### 主要セクション

#### 2.1 実験追跡設定 (experiment_tracking)

MLflowによる実験追跡の設定です。

```yaml
experiment_tracking:
  enabled: true
  tracking_uri: null
  experiment_name: "neuralforecast_optimization"
  auto_logging:
    enabled: true
    log_models: true
  metrics:
    primary_metric: "val_loss"
```

**使用例:**
```python
import yaml
import mlflow

# 設定を読み込む
with open('configs/default_configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

tracking_config = config['experiment_tracking']

# MLflowの設定
if tracking_config['enabled']:
    if tracking_config['tracking_uri']:
        mlflow.set_tracking_uri(tracking_config['tracking_uri'])
    
    mlflow.set_experiment(tracking_config['experiment_name'])
```

#### 2.2 ハイパーパラメータ最適化設定 (hyperparameter_optimization)

Optuna/Ray Tuneの最適化設定です。

```yaml
hyperparameter_optimization:
  backend: "optuna"
  num_samples: null  # 自動推奨
  pruning:
    enabled: true
    patience: 10
  optuna:
    sampler:
      default: "TPESampler"
      n_startup_trials: 10
```

**使用例:**
```python
import optuna

# 設定を読み込む
opt_config = config['hyperparameter_optimization']['optuna']

# Optunaのサンプラーを作成
if opt_config['sampler']['default'] == 'TPESampler':
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=opt_config['sampler']['n_startup_trials'],
        multivariate=opt_config['sampler']['multivariate'],
        seed=opt_config['sampler']['seed']
    )
```

#### 2.3 分散計算設定 (distributed_computing)

Ray分散実行の設定です。

```yaml
distributed_computing:
  resources:
    cpus_per_trial: 4
    gpus_per_trial: 1
  concurrent_trials: 4
  ray_init:
    include_dashboard: true
```

**使用例:**
```python
import ray
from ray import tune

# 設定を読み込む
ray_config = config['distributed_computing']

# Ray初期化
if not ray.is_initialized():
    ray.init(
        num_cpus=ray_config['ray_init']['num_cpus'],
        num_gpus=ray_config['ray_init']['num_gpus'],
        include_dashboard=ray_config['ray_init']['include_dashboard']
    )

# リソース設定
resources = {
    "cpu": ray_config['resources']['cpus_per_trial'],
    "gpu": ray_config['resources']['gpus_per_trial']
}
```

#### 2.4 モデル訓練設定 (model_training)

モデルの訓練に関するデフォルト設定です。

```yaml
model_training:
  common:
    max_steps: 1000
    learning_rate: 0.001
    batch_size: 32
  optimizer:
    type: "Adam"
    weight_decay: 0.0
```

**使用例:**
```python
# 設定を読み込む
train_config = config['model_training']['common']

# ハイパーパラメータ探索空間を生成
from ray import tune

search_space = {
    'max_steps': tune.choice(
        config['default_hyperparameter_space']['common']['max_steps']['values']
    ),
    'learning_rate': tune.loguniform(
        config['default_hyperparameter_space']['common']['learning_rate']['low'],
        config['default_hyperparameter_space']['common']['learning_rate']['high']
    ),
    'batch_size': tune.choice(
        config['default_hyperparameter_space']['common']['batch_size']['values']
    )
}
```

#### 2.5 探索アルゴリズム設定 (search_algorithm)

アルゴリズム選択とプルーニング戦略の設定です。

```yaml
search_algorithm:
  num_samples_recommendation:
    complex_model:
      large_data: 200
  algorithm_selection:
    large_data:
      complex_model: "TPESampler_multivariate"
```

**使用例:**
```python
def get_recommended_samples(
    model_complexity: str,
    dataset_size: str,
    config: dict
) -> int:
    """推奨試行回数を取得"""
    recommendation = config['search_algorithm']['num_samples_recommendation']
    return recommendation[model_complexity][dataset_size]

# 使用例
num_samples = get_recommended_samples(
    model_complexity='complex_model',
    dataset_size='large_data',
    config=config
)
print(f"Recommended samples: {num_samples}")
```

### 設定のカスタマイズ

#### 方法1: YAMLファイルを直接編集

```yaml
# custom_config.yaml
experiment_tracking:
  enabled: true
  tracking_uri: "http://mlflow-server:5000"
  experiment_name: "production_forecast"

hyperparameter_optimization:
  backend: "optuna"
  num_samples: 100
```

```python
import yaml

# カスタム設定を読み込む
with open('custom_config.yaml', 'r') as f:
    custom_config = yaml.safe_load(f)

# デフォルト設定をマージ
with open('configs/default_configs.yaml', 'r') as f:
    default_config = yaml.safe_load(f)

# 設定をマージ（カスタム設定が優先）
from copy import deepcopy

merged_config = deepcopy(default_config)
for key, value in custom_config.items():
    if isinstance(value, dict) and key in merged_config:
        merged_config[key].update(value)
    else:
        merged_config[key] = value
```

#### 方法2: Pythonで動的に上書き

```python
import yaml

# デフォルト設定を読み込む
with open('configs/default_configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 特定の設定を上書き
config['hyperparameter_optimization']['num_samples'] = 50
config['distributed_computing']['resources']['gpus_per_trial'] = 2
config['experiment_tracking']['experiment_name'] = 'my_experiment'

# 更新された設定を保存
with open('my_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
```

---

## 3. 統合使用例

### 例1: 基本的な使用

```python
import yaml
from auto_model_factory import create_auto_model

# 設定ファイルを読み込む
with open('configs/default_configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open('configs/model_characteristics.yaml', 'r') as f:
    model_chars = yaml.safe_load(f)

# モデル選択
model_name = "AutoTFT"
model_char = model_chars[model_name]

# 推奨試行回数を取得
complexity = model_char['complexity']
data_size = 'large_data'  # データセットサイズに応じて設定
num_samples = config['search_algorithm']['num_samples_recommendation'][
    f'{complexity}_model'
][data_size]

# モデル作成
auto_model = create_auto_model(
    model_name=model_char['base_model_name'],
    h=24,
    dataset=df,
    backend=config['hyperparameter_optimization']['backend'],
    num_samples=num_samples,
    use_mlflow=config['experiment_tracking']['enabled'],
    verbose=True
)
```

### 例2: 複数モデルの比較

```python
import yaml
import pandas as pd
from auto_model_factory import create_auto_model

# 設定読み込み
with open('configs/model_characteristics.yaml', 'r') as f:
    model_chars = yaml.safe_load(f)

# 中規模の複雑度のモデルを選択
moderate_models = [
    name for name, char in model_chars.items()
    if char['complexity'] == 'moderate' and 
    char['exogenous']['support'] != 'none'
]

results = {}
for model_name in moderate_models[:3]:  # 上位3モデルで比較
    print(f"\nOptimizing {model_name}...")
    
    model_char = model_chars[model_name]
    
    auto_model = create_auto_model(
        model_name=model_char['base_model_name'],
        h=24,
        dataset=df,
        num_samples=50,
        backend='optuna'
    )
    
    predictions = auto_model.predict(dataset=df)
    results[model_name] = predictions

# 結果の比較
for model_name, pred in results.items():
    print(f"{model_name}: {pred.head()}")
```

### 例3: プロダクション環境向け設定

```python
import yaml

# デフォルト設定を読み込む
with open('configs/default_configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

# プロダクション向けにカスタマイズ
production_config = config.copy()
production_config.update({
    'experiment_tracking': {
        'enabled': True,
        'tracking_uri': 'http://mlflow-prod:5000',
        'experiment_name': 'production_forecast_v2'
    },
    'hyperparameter_optimization': {
        'backend': 'optuna',
        'num_samples': 100,
        'pruning': {
            'enabled': True,
            'patience': 5
        }
    },
    'distributed_computing': {
        'resources': {
            'cpus_per_trial': 8,
            'gpus_per_trial': 2
        },
        'concurrent_trials': 4
    },
    'logging': {
        'level': 'INFO',
        'handlers': {
            'file': {
                'enabled': True,
                'filename': 'production_optimization.log'
            }
        }
    }
})

# プロダクション設定を保存
with open('production_config.yaml', 'w') as f:
    yaml.dump(production_config, f, default_flow_style=False)
```

---

## 4. ベストプラクティス

### 4.1 設定管理

1. **バージョン管理**: 設定ファイルはGitで管理する
2. **環境分離**: dev/staging/prod環境ごとに設定ファイルを用意
3. **機密情報**: API keyやパスワードは環境変数で管理

```python
import os
import yaml

# 環境変数から機密情報を読み込む
config['experiment_tracking']['tracking_uri'] = os.getenv(
    'MLFLOW_TRACKING_URI',
    config['experiment_tracking']['tracking_uri']
)
```

### 4.2 設定の検証

```python
import yaml
from jsonschema import validate

# スキーマ定義（例）
config_schema = {
    "type": "object",
    "properties": {
        "hyperparameter_optimization": {
            "type": "object",
            "properties": {
                "backend": {"enum": ["optuna", "ray"]},
                "num_samples": {"type": ["integer", "null"]}
            }
        }
    }
}

# 設定を検証
with open('configs/default_configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

try:
    validate(instance=config, schema=config_schema)
    print("Configuration is valid")
except Exception as e:
    print(f"Configuration error: {e}")
```

### 4.3 設定の継承

```python
import yaml

def load_config_with_inheritance(config_files: list) -> dict:
    """複数の設定ファイルを継承してマージ"""
    merged_config = {}
    
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # 深いマージ
        def deep_merge(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(merged_config, config)
    
    return merged_config

# 使用例
config = load_config_with_inheritance([
    'configs/default_configs.yaml',
    'configs/team_config.yaml',
    'configs/my_config.yaml'
])
```

---

## 5. トラブルシューティング

### よくある問題

#### 問題1: YAMLの構文エラー

```python
import yaml

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except yaml.YAMLError as e:
    print(f"YAML syntax error: {e}")
    # エラー箇所を確認
```

#### 問題2: 設定値の型が不正

```python
def validate_config_types(config: dict) -> bool:
    """設定値の型を検証"""
    try:
        # 数値チェック
        assert isinstance(
            config['hyperparameter_optimization']['num_samples'],
            (int, type(None))
        )
        
        # 列挙型チェック
        assert config['hyperparameter_optimization']['backend'] in ['optuna', 'ray']
        
        return True
    except (AssertionError, KeyError) as e:
        print(f"Configuration validation error: {e}")
        return False
```

#### 問題3: 設定ファイルが見つからない

```python
from pathlib import Path

def find_config_file(filename: str, search_paths: list) -> Path:
    """設定ファイルを検索"""
    for search_path in search_paths:
        config_path = Path(search_path) / filename
        if config_path.exists():
            return config_path
    
    raise FileNotFoundError(
        f"Config file '{filename}' not found in {search_paths}"
    )

# 使用例
config_path = find_config_file(
    'default_configs.yaml',
    ['./configs', '../configs', '/etc/neuralforecast']
)
```

---

## 6. まとめ

この設定ファイル群により、以下が実現できます：

1. **一元管理**: 全モデルとシステム設定を単一ファイルで管理
2. **柔軟性**: 環境や用途に応じた設定のカスタマイズ
3. **再現性**: 設定ファイルで実験を完全に再現可能
4. **可読性**: YAMLフォーマットで人間が読みやすい
5. **保守性**: 設定の変更が容易で影響範囲が明確

これらの設定ファイルを適切に活用することで、効率的かつ堅牢な時系列予測システムを構築できます。
