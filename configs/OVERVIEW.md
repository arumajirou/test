# NeuralForecast 自動最適化システム - 設定ファイル概要

## 📋 作成されたファイル

以下の3つのファイルが作成されました：

1. **model_characteristics.yaml** (20KB)
   - 全28種類のAutoModelsの特性定義
   
2. **default_configs.yaml** (11KB)
   - システム全体のデフォルト設定（12セクション）
   
3. **configs_README.md** (18KB)
   - 詳細な使用方法ガイド

---

## 🎯 各ファイルの目的

### 1. model_characteristics.yaml

**目的**: モデル特性の一元管理

**定義内容**:
- ✅ 28モデル全ての詳細特性
- ✅ パラメータタイプ（連続/離散/カテゴリカル）
- ✅ 最適化タイプとドロップアウト設定
- ✅ スケーラーと外生変数サポート
- ✅ バックエンド互換性

**モデル分類**:
```
Transformer系  (7モデル): TFT, Autoformer, PatchTST, etc.
RNN系         (6モデル): LSTM, GRU, TCN, DeepAR, etc.
Linear系      (4モデル): DLinear, NLinear, MLP, etc.
NBEATS系      (3モデル): NBEATS, NBEATSx, NHITS
その他        (8モデル): TiDE, TimeMixer, StemGNN, etc.
```

**使用例**:
```python
import yaml

with open('model_characteristics.yaml', 'r') as f:
    models = yaml.safe_load(f)

# モデル情報の取得
tft = models['AutoTFT']
print(f"Complexity: {tft['complexity']}")
print(f"Dropout: {tft['optimization']['dropout']['default']}")
```

---

### 2. default_configs.yaml

**目的**: システム全体の設定管理

**12の主要セクション**:

1. **experiment_tracking** - MLflow実験追跡
2. **hyperparameter_optimization** - Optuna/Ray設定
3. **distributed_computing** - Ray分散実行
4. **validation** - 検証ルール
5. **logging** - ログ設定
6. **model_training** - 訓練パラメータ
7. **search_algorithm** - アルゴリズム選択
8. **default_hyperparameter_space** - 探索空間定義
9. **performance** - パフォーマンス設定
10. **security** - セキュリティ設定
11. **debug** - デバッグ設定
12. **system** - システム全般設定

**主要機能**:
```yaml
# 自動推奨試行回数
search_algorithm:
  num_samples_recommendation:
    complex_model:
      large_data: 200  # 大規模データ×複雑モデル

# アルゴリズム自動選択
algorithm_selection:
  large_data:
    complex_model: "TPESampler_multivariate"

# デフォルト探索空間
default_hyperparameter_space:
  common:
    learning_rate:
      type: "loguniform"
      low: 1e-5
      high: 1e-2
```

**使用例**:
```python
import yaml

with open('default_configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

# MLflow設定
mlflow_config = config['experiment_tracking']

# 推奨試行回数
samples = config['search_algorithm']['num_samples_recommendation']
```

---

## 🚀 クイックスタート

### ステップ1: ファイルの配置

```bash
# プロジェクトルートにディレクトリ作成
mkdir -p configs

# YAMLファイルを配置
mv model_characteristics.yaml configs/
mv default_configs.yaml configs/
```

### ステップ2: 基本的な使用

```python
import yaml
import pandas as pd
from auto_model_factory import create_auto_model

# 設定ファイル読み込み
with open('configs/model_characteristics.yaml', 'r') as f:
    model_chars = yaml.safe_load(f)

with open('configs/default_configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

# データ準備
df = pd.read_csv('data.csv')

# モデル選択と最適化
model_name = "AutoNHITS"
model_info = model_chars[model_name]

# 推奨試行回数を取得
complexity = model_info['complexity']
num_samples = config['search_algorithm']['num_samples_recommendation'][
    f'{complexity}_model']['medium_data']

# モデル作成
auto_model = create_auto_model(
    model_name=model_info['base_model_name'],
    h=24,
    dataset=df,
    num_samples=num_samples,
    backend=config['hyperparameter_optimization']['backend'],
    use_mlflow=config['experiment_tracking']['enabled']
)

# 予測
predictions = auto_model.predict(dataset=df)
```

### ステップ3: カスタマイズ

```python
# デフォルト設定を読み込み
with open('configs/default_configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 特定の設定を上書き
config['hyperparameter_optimization']['num_samples'] = 100
config['experiment_tracking']['experiment_name'] = 'my_experiment'
config['distributed_computing']['resources']['gpus_per_trial'] = 2

# カスタム設定を保存
with open('my_custom_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
```

---

## 📊 設定ファイルの構造

```
configs/
├── model_characteristics.yaml
│   ├── AutoAutoformer          # Transformer系
│   ├── AutoTFT                 # (複雑度: complex)
│   ├── AutoPatchTST
│   ├── ...
│   ├── AutoLSTM                # RNN系
│   ├── AutoGRU                 # (複雑度: moderate)
│   ├── ...
│   ├── AutoDLinear             # Linear系
│   ├── AutoNLinear             # (複雑度: simple)
│   ├── ...
│   └── AutoNHITS               # NBEATS系
│
└── default_configs.yaml
    ├── experiment_tracking      # MLflow
    ├── hyperparameter_optimization  # Optuna/Ray
    ├── distributed_computing    # Ray
    ├── validation              # 検証
    ├── logging                 # ログ
    ├── model_training          # 訓練
    ├── search_algorithm        # アルゴリズム選択
    ├── default_hyperparameter_space  # 探索空間
    ├── performance             # パフォーマンス
    ├── security                # セキュリティ
    ├── debug                   # デバッグ
    └── system                  # システム
```

---

## 🎓 高度な使用例

### 例1: モデルフィルタリング

```python
# 外生変数対応の中規模モデルを検索
suitable_models = [
    name for name, char in model_chars.items()
    if char['complexity'] == 'moderate' and
       char['exogenous']['support'] == 'full'
]
print(f"Suitable models: {suitable_models}")
```

### 例2: 環境別設定

```python
import os

# 環境変数でMLflow URIを設定
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
config['experiment_tracking']['tracking_uri'] = mlflow_uri

# GPU数を環境に応じて調整
n_gpus = int(os.getenv('N_GPUS', '1'))
config['distributed_computing']['resources']['gpus_per_trial'] = n_gpus
```

### 例3: 複数モデルの自動比較

```python
# 同じカテゴリの全モデルで比較
category = 'nbeats'
models_to_test = [
    name for name, char in model_chars.items()
    if char['model_category'] == category
]

results = {}
for model_name in models_to_test:
    model_info = model_chars[model_name]
    auto_model = create_auto_model(
        model_name=model_info['base_model_name'],
        h=24,
        dataset=df,
        num_samples=50
    )
    results[model_name] = auto_model.predict(dataset=df)
```

---

## ✅ チェックリスト

設定ファイルを使用する前に確認：

- [ ] YAMLファイルがプロジェクトのconfigsディレクトリに配置されている
- [ ] Python環境にPyYAMLがインストールされている (`pip install pyyaml`)
- [ ] 既存のmodel_characteristics.pyとの整合性を確認
- [ ] 環境変数で機密情報を管理
- [ ] カスタム設定ファイルをバージョン管理に含める
- [ ] 本番環境用の設定を別ファイルで管理

---

## 🔗 関連ドキュメント

詳細な使用方法は **configs_README.md** を参照してください：

- モデル特性の詳細説明
- 全12セクションの設定項目
- 実践的な使用例（10以上）
- トラブルシューティングガイド
- ベストプラクティス

---

## 📞 サポート

問題が発生した場合：

1. **configs_README.md** のトラブルシューティングセクションを確認
2. YAMLの構文エラーは `yaml.safe_load()` でチェック
3. 設定値の型は `isinstance()` で検証
4. 不明なモデル名は `model_chars.keys()` で確認

---

## 🎉 次のステップ

1. ✅ 設定ファイルをプロジェクトに配置
2. ✅ 基本的な使用例を試す
3. ✅ 自分の用途に合わせてカスタマイズ
4. ✅ プロダクション環境向けの設定を作成
5. ✅ 実験結果をMLflowで追跡

この設定ファイル群により、NeuralForecastの自動最適化システムを
効率的かつ柔軟に運用できます！
