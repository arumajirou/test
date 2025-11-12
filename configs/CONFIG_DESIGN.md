# NeuralForecast AutoModels 設定ファイル設計書

## 概要

このドキュメントは、NeuralForecast AutoModelsフレームワークの設定ファイル構造と設計思想を説明します。

---

## ディレクトリ構造

```
configs/
├── model_characteristics.yaml   # モデル特性定義（YAML）
└── default_configs.yaml         # デフォルト設定
```

---

## 1. model_characteristics.yaml

### 目的

全AutoModelsの特性を一元管理し、以下の目的を実現します：

1. **検索アルゴリズムの自動選択**：モデル特性に基づき最適な探索手法を選択
2. **パラメータ設定の標準化**：各モデルのデフォルト値を統一管理
3. **外生変数サポートの明確化**：モデルごとの対応状況を可視化
4. **探索空間の複雑度推定**：最適化戦略の決定に活用

### 主要セクション

#### 1.1 列挙型定義

```yaml
optimization_types:
  continuous: "continuous"          # 連続パラメータのみ
  discrete: "discrete"              # 離散パラメータのみ
  mixed: "mixed"                    # 混合探索空間
  multi_objective: "multi_objective"  # 多目的最適化

exogenous_support_types:
  none: "none"           # 外生変数非対応
  futr_only: "futr_only"  # 未来外生のみ
  hist_only: "hist_only"  # 履歴外生のみ
  stat_only: "stat_only"  # 静的外生のみ
  full: "full"           # 全種類対応
```

**設計意図**：
- Python Enumと同期させ、型安全性を保証
- 外生変数サポートの細かい分類により、適切なモデル選択を支援

#### 1.2 モデル特性定義

各モデルには以下の属性を定義：

| 属性 | 型 | 説明 | 例 |
|------|------|------|------|
| `model_name` | string | モデル名 | "AutoTFT" |
| `base_model_name` | string | ベースモデル名 | "TFT" |
| `has_continuous_params` | bool | 連続パラメータの有無 | true |
| `has_discrete_params` | bool | 離散パラメータの有無 | true |
| `has_categorical_params` | bool | カテゴリカルパラメータの有無 | true |
| `supports_multi_objective` | bool | 多目的最適化対応 | false |
| `typical_search_space_size` | int | 典型的な探索空間サイズ | 20000 |
| `dropout_supported` | bool | ドロップアウト対応 | true |
| `dropout_key` | string/null | ドロップアウトパラメータ名 | "dropout" |
| `dropout_default` | float/null | ドロップアウトデフォルト値 | 0.1 |
| `default_scaler` | string | デフォルトスケーラー | "identity" |
| `scaler_choices` | list | 選択可能なスケーラー | ["identity", "standard"] |
| `exogenous_support` | string | 外生変数サポート種別 | "full" |
| `early_stop_default` | int | 早期停止デフォルト値 | -1 (無効) |
| `backend_choices` | list | 対応バックエンド | ["ray", "optuna"] |
| `pruner_default_enabled_optuna` | bool | Optunaでprunerデフォルト有効 | true |
| `notes` | string | 備考・特記事項 | "説明文" |

**設計意図**：
- `typical_search_space_size`：探索戦略決定の重要指標
- `dropout_*`フィールド：正則化パラメータの統一管理
- `exogenous_support`：データ要件の明確化
- `notes`：モデル選択時の参考情報

#### 1.3 モデル複雑度分類

```yaml
model_complexity:
  simple:      # 単純なモデル（MLP, Linear等）
    - "AutoMLP"
    - "AutoNLinear"
    - "AutoDLinear"
  
  moderate:    # 中程度の複雑度（RNN, NHITS等）
    - "AutoNHITS"
    - "AutoTCN"
    - "AutoGRU"
    
  complex:     # 複雑なモデル（Transformer系）
    - "AutoTFT"
    - "AutoAutoformer"
    - "AutoPatchTST"
```

**設計意図**：
- 計算コスト推定の基準
- 試行回数・タイムアウト設定の自動調整
- ユーザーへの推奨モデル提示

---

## 2. default_configs.yaml

### 目的

フレームワーク全体のデフォルト設定を一元管理し、以下を実現：

1. **統一的な設定管理**：MLflow、Optuna、Rayの設定を一箇所に集約
2. **環境依存設定の分離**：開発/本番環境の切り替えを容易に
3. **ベストプラクティスの標準化**：推奨設定をデフォルトとして提供
4. **カスタマイズの柔軟性**：すべての設定をオーバーライド可能

### 主要セクション

#### 2.1 MLflow統合設定

```yaml
mlflow:
  tracking_uri: "file:./mlruns"
  default_experiment_name: "neuralforecast_auto"
  
  auto_logging:
    enabled: true
    log_models: true
    log_params: true
    log_metrics: true
```

**設計意図**：
- **tracking_uri**: ローカル/リモート/Databricksの切り替えが容易
- **auto_logging**: デフォルトで全自動ログを有効化（実験再現性の確保）
- **model_registry**: 本番運用時のモデル管理を想定

#### 2.2 Optuna最適化設定

```yaml
optuna:
  sampler:
    default: "TPESampler"
    tpe:
      n_startup_trials: 10
      n_ei_candidates: 24
      multivariate: false
  
  pruner:
    enabled: true
    default: "MedianPruner"
    median:
      n_startup_trials: 5
      n_warmup_steps: 10
```

**設計意図**：
- **TPESampler**: 混合探索空間に対応した汎用性の高いアルゴリズム
- **MedianPruner**: 早期に低性能な試行を打ち切り、効率化
- **n_startup_trials**: 初期ランダムサンプリングで探索を開始
- **multivariate**: デフォルトfalse（計算コスト考慮）、必要時にtrue化

#### 2.3 Ray Tune設定

```yaml
ray:
  search_algorithm:
    default: "OptunaSearch"
  
  scheduler:
    default: "ASHAScheduler"
    asha:
      max_t: 100
      grace_period: 10
      reduction_factor: 3
```

**設計意図**：
- **OptunaSearch**: Rayバックエンドでも Optunaの強力なサンプラーを活用
- **ASHAScheduler**: 効率的な早期停止により計算リソースを節約
- **grace_period**: 最低限の学習期間を保証（過度な早期停止を防ぐ）

#### 2.4 検索アルゴリズム自動選択設定

```yaml
search_algorithm_selector:
  backend: "optuna"
  
  search_complexity:
    low_threshold: 5      # < 5パラメータ
    medium_threshold: 15  # 5-15パラメータ
  
  dataset_size:
    small_threshold: 1000
    large_threshold: 100000
  
  recommended_samples:
    low_simple: 10
    medium_moderate: 30
    high_complex: 150
```

**設計意図**：
- **閾値設定**: 経験則に基づく妥当な境界値
- **推奨試行回数**: 探索空間×モデル複雑度のマトリクスで定義
- **データセットサイズ補正**: 小規模データでは試行回数を削減、大規模では増加

#### 2.5 ロギング設定

```yaml
logging:
  level: "INFO"
  format: "json"
  
  handlers:
    console:
      enabled: true
      level: "INFO"
    
    file:
      enabled: true
      level: "DEBUG"
      filename: "logs/neuralforecast_auto.log"
      max_bytes: 10485760  # 10MB
      backup_count: 5
```

**設計意図**：
- **JSONフォーマット**: 構造化ログで後処理・分析が容易
- **ファイルローテーション**: ディスク容量管理
- **モジュール別ログレベル**: 外部ライブラリのログを抑制

#### 2.6 検証設定

```yaml
validation:
  holdout:
    enabled: true
    test_size: 0.2
  
  metrics:
    primary: "mae"
    additional:
      - "mse"
      - "rmse"
      - "mape"
  
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.0001
```

**設計意図**：
- **ホールドアウト検証**: デフォルトで有効（CVはオプション）
- **複数メトリクス**: 包括的な評価のため補助メトリクスも記録
- **早期停止**: 過学習防止とリソース節約

#### 2.7 モデル学習設定

```yaml
training:
  max_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
  loss: "mae"
  
  accelerator: "auto"  # GPU自動検出
  devices: 1
```

**設計意図**：
- **保守的なデフォルト**: 多くのケースで安定した学習が可能
- **accelerator: auto**: 環境に応じた自動最適化
- **adam**: 汎用性の高いオプティマイザー

#### 2.8 実験管理設定

```yaml
experiment:
  auto_naming:
    enabled: true
    pattern: "{model}_{timestamp}"
  
  auto_tagging:
    enabled: true
    include_git_info: false
    include_environment: true
  
  results:
    save_predictions: true
    save_checkpoints: true
    output_dir: "outputs"
```

**設計意図**：
- **自動命名**: 実験の一意性と追跡性を確保
- **自動タグ付け**: 実験メタデータの自動収集
- **結果保存**: 再現性と後処理のための完全な保存

---

## 3. 設定ファイルの使用方法

### 3.1 Python側での読み込み

```python
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "configs/default_configs.yaml") -> Dict[str, Any]:
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model_characteristics(
    config_path: str = "configs/model_characteristics.yaml"
) -> Dict[str, Any]:
    """モデル特性を読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 使用例
config = load_config()
mlflow_config = config['mlflow']
optuna_config = config['optuna']

model_chars = load_model_characteristics()
tft_characteristics = model_chars['models']['AutoTFT']
```

### 3.2 設定のオーバーライド

```python
# デフォルト設定をロード
config = load_config()

# 特定の設定をオーバーライド
config['mlflow']['tracking_uri'] = "http://mlflow-server:5000"
config['optuna']['optimization']['n_trials'] = 100

# カスタム設定で実行
tracker = MLflowTracker(
    tracking_uri=config['mlflow']['tracking_uri'],
    experiment_name=config['mlflow']['default_experiment_name']
)
```

### 3.3 環境別設定の管理

```yaml
# configs/production.yaml
mlflow:
  tracking_uri: "databricks"
  default_experiment_name: "neuralforecast_production"
  
optuna:
  optimization:
    n_trials: 200
    n_jobs: 8
    
logging:
  level: "WARNING"
```

```python
# 環境に応じた設定読み込み
import os

env = os.getenv("ENV", "development")
config_file = f"configs/{env}.yaml"

if Path(config_file).exists():
    config = load_config(config_file)
else:
    config = load_config("configs/default_configs.yaml")
```

---

## 4. 設計思想

### 4.1 YAML形式の選択理由

1. **可読性**: 人間が読みやすく、編集しやすい
2. **階層構造**: ネストした設定を直感的に表現
3. **コメント対応**: 設定の意図を文書化できる
4. **型サポート**: bool, int, float, list, dictを自然に表現
5. **エコシステム**: 多くのツールが標準サポート

### 4.2 設定の粒度

- **粗すぎる設定**: カスタマイズが困難
- **細かすぎる設定**: 設定ファイルが肥大化し管理が困難

→ **バランスの取れた粒度**：
- 頻繁に変更される設定は明示的に分離
- 固定的な設定はコードに埋め込み
- 環境依存の設定は環境変数で制御可能に

### 4.3 デフォルト値の哲学

1. **安全第一**: 破壊的な動作をデフォルトにしない
2. **一般性**: 多様なユースケースで動作する設定
3. **効率性**: 計算リソースを無駄にしない
4. **再現性**: 同じ設定で同じ結果が得られる

### 4.4 拡張性

- 新しいモデル追加時: `model_characteristics.yaml`に1エントリ追加
- 新しいバックエンド追加時: `default_configs.yaml`に新セクション追加
- 後方互換性: 古い設定ファイルも動作するようエラーハンドリング

---

## 5. ベストプラクティス

### 5.1 設定ファイルのバージョン管理

```yaml
# configs/default_configs.yaml の冒頭
config_version: "1.0.0"
last_updated: "2025-01-15"
```

### 5.2 設定の検証

```python
from jsonschema import validate

def validate_config(config: Dict[str, Any], schema_path: str) -> bool:
    """設定ファイルをスキーマで検証"""
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    try:
        validate(instance=config, schema=schema)
        return True
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return False
```

### 5.3 設定のマージ

```python
def merge_configs(base: Dict, override: Dict) -> Dict:
    """設定を再帰的にマージ"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result

# 使用例
base_config = load_config("configs/default_configs.yaml")
prod_config = load_config("configs/production.yaml")
final_config = merge_configs(base_config, prod_config)
```

---

## 6. トラブルシューティング

### 6.1 よくある問題

**問題1**: YAML解析エラー

```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**解決策**: インデントを確認（スペース2個で統一）

---

**問題2**: 設定が反映されない

**解決策**: 
1. ファイルパスが正しいか確認
2. キャッシュをクリア
3. デバッグログで読み込み状況を確認

---

**問題3**: 型エラー

```python
TypeError: 'NoneType' object is not subscriptable
```

**解決策**: デフォルト値を設定

```python
config.get('mlflow', {}).get('tracking_uri', 'file:./mlruns')
```

---

## 7. 今後の拡張予定

- [ ] JSON Schema定義の追加（バリデーション強化）
- [ ] 設定ファイルのテンプレート生成ツール
- [ ] GUI設定エディタの提供
- [ ] 設定差分の可視化ツール
- [ ] クラウド環境向けプリセット設定

---

## 付録：関連ファイル

- `model_characteristics.py`: Python実装版
- `search_algorithm_selector.py`: 自動選択ロジック
- `mlflow_integration.py`: MLflow統合
- `logging_config.py`: ロギング設定
- `validation.py`: 検証ロジック

---

**更新履歴**

- 2025-01-15: 初版作成
- 設定ファイルv1.0.0リリース
