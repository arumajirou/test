# 自動モデルファクトリー - トラブルシューティング

## 📗 目次

- [クイック診断チェックリスト](#クイック診断チェックリスト)
- [インストールとセットアップの問題](#インストールとセットアップの問題)
- [データ関連の問題](#データ関連の問題)
- [メモリとパフォーマンスの問題](#メモリとパフォーマンスの問題)
- [最適化の問題](#最適化の問題)
- [MLflowの問題](#mlflowの問題)
- [GPU関連の問題](#gpu関連の問題)
- [エラーメッセージ別ガイド](#エラーメッセージ別ガイド)
- [デバッグ手法](#デバッグ手法)
- [FAQ](#faq)

---

## クイック診断チェックリスト

問題に直面したら、まず以下をチェックしてください：

```python
# diagnostic_check.py - 包括的な診断スクリプト
import sys
import torch
import pandas as pd
import numpy as np

def run_diagnostics():
    """システム診断を実行"""
    
    print("=" * 80)
    print("🔍 SYSTEM DIAGNOSTICS")
    print("=" * 80)
    
    # 1. Python環境
    print("\n1. Python Environment:")
    print(f"   Python version: {sys.version}")
    print(f"   Python executable: {sys.executable}")
    
    # 2. 必須パッケージ
    print("\n2. Required Packages:")
    try:
        import neuralforecast
        print(f"   ✅ neuralforecast: {neuralforecast.__version__}")
    except ImportError:
        print("   ❌ neuralforecast: NOT INSTALLED")
    
    try:
        import optuna
        print(f"   ✅ optuna: {optuna.__version__}")
    except ImportError:
        print("   ❌ optuna: NOT INSTALLED")
    
    try:
        import ray
        print(f"   ✅ ray: {ray.__version__}")
    except ImportError:
        print("   ❌ ray: NOT INSTALLED")
    
    try:
        import mlflow
        print(f"   ✅ mlflow: {mlflow.__version__}")
    except ImportError:
        print("   ❌ mlflow: NOT INSTALLED")
    
    # 3. PyTorch とCUDA
    print("\n3. PyTorch & CUDA:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # 4. システムリソース
    print("\n4. System Resources:")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"   Total RAM: {mem.total / 1e9:.2f} GB")
        print(f"   Available RAM: {mem.available / 1e9:.2f} GB")
        print(f"   CPU count: {psutil.cpu_count()}")
        disk = psutil.disk_usage('/')
        print(f"   Disk free: {disk.free / 1e9:.2f} GB")
    except ImportError:
        print("   ⚠️  psutil not installed (optional)")
    
    # 5. プロジェクトモジュール
    print("\n5. Project Modules:")
    modules = [
        'auto_model_factory',
        'validation',
        'search_algorithm_selector',
        'model_characteristics',
        'mlflow_integration',
        'logging_config'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
    
    print("\n" + "=" * 80)
    print("✅ DIAGNOSTICS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_diagnostics()
```

**使用方法**:
```bash
python diagnostic_check.py
```

---

## インストールとセットアップの問題

### 問題: パッケージのインポートエラー

#### 症状
```python
ImportError: No module named 'neuralforecast'
ImportError: No module named 'optuna'
```

#### 解決策

**ステップ1**: 仮想環境の確認
```bash
# 仮想環境が有効になっているか確認
which python
# または
python --version
```

**ステップ2**: パッケージの再インストール
```bash
# 基本パッケージ
pip install --upgrade neuralforecast
pip install --upgrade optuna
pip install --upgrade 'ray[tune]'
pip install --upgrade mlflow
pip install --upgrade pytorch-lightning

# GPU版PyTorch（GPU使用時）
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**ステップ3**: 依存関係の確認
```bash
pip check
```

---

### 問題: CUDA/GPU認識エラー

#### 症状
```python
RuntimeError: CUDA out of memory
RuntimeError: No CUDA GPUs are available
```

#### 解決策

**診断スクリプト**:
```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Available memory: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1e9:.2f} GB")
else:
    print("⚠️ CUDA not available")
    print("Solution:")
    print("1. Check NVIDIA driver: nvidia-smi")
    print("2. Reinstall CUDA-compatible PyTorch")
    print("3. Or use CPU-only mode: gpus=0")
```

**解決法1**: ドライバーの確認
```bash
# NVIDIA ドライバーとCUDAの確認
nvidia-smi

# CUDAバージョンに合ったPyTorchをインストール
# CUDA 11.8の場合:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**解決法2**: CPU-onlyモードで実行
```python
auto_model = create_auto_model(
    model_name="NHITS",
    h=7,
    dataset=df,
    gpus=0  # GPUを使用しない
)
```

---

## データ関連の問題

### 問題: データセット検証エラー

#### 症状
```
ValidationError: Missing required column: 'ds'
ValidationError: Invalid date format in column 'ds'
ValidationError: Missing values found in column 'y'
```

#### 解決策

**診断と修正スクリプト**:
```python
import pandas as pd
import numpy as np
from datetime import datetime

def diagnose_and_fix_dataset(df, fix=False):
    """データセットの問題を診断し、オプションで修正"""
    
    print("🔍 Diagnosing dataset...\n")
    
    issues = []
    fixes = {}
    
    # 1. 必須カラムのチェック
    required_cols = ['unique_id', 'ds', 'y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
    else:
        print("✅ All required columns present")
    
    # 2. データ型のチェック
    if 'ds' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            issues.append("Column 'ds' is not datetime type")
            print("❌ 'ds' column is not datetime")
            print(f"   Current type: {df['ds'].dtype}")
            
            if fix:
                try:
                    df['ds'] = pd.to_datetime(df['ds'])
                    fixes['ds_converted'] = True
                    print("   ✅ Fixed: Converted to datetime")
                except Exception as e:
                    print(f"   ⚠️  Could not convert: {e}")
        else:
            print("✅ 'ds' column is datetime type")
    
    if 'y' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['y']):
            issues.append("Column 'y' is not numeric")
            print("❌ 'y' column is not numeric")
            print(f"   Current type: {df['y'].dtype}")
            
            if fix:
                try:
                    df['y'] = pd.to_numeric(df['y'])
                    fixes['y_converted'] = True
                    print("   ✅ Fixed: Converted to numeric")
                except Exception as e:
                    print(f"   ⚠️  Could not convert: {e}")
        else:
            print("✅ 'y' column is numeric type")
    
    # 3. 欠損値のチェック
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        issues.append(f"Missing values found")
        print(f"\n❌ Missing values detected:")
        for col, count in null_cols.items():
            print(f"   {col}: {count} ({count/len(df)*100:.2f}%)")
        
        if fix:
            df_before = len(df)
            df = df.dropna()
            df_after = len(df)
            fixes['rows_dropped'] = df_before - df_after
            print(f"   ✅ Fixed: Dropped {df_before - df_after} rows with missing values")
    else:
        print("\n✅ No missing values")
    
    # 4. 重複のチェック
    if 'unique_id' in df.columns and 'ds' in df.columns:
        duplicates = df.duplicated(subset=['unique_id', 'ds']).sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate records")
            print(f"\n❌ Found {duplicates} duplicate records")
            
            if fix:
                df = df.drop_duplicates(subset=['unique_id', 'ds'])
                fixes['duplicates_removed'] = duplicates
                print(f"   ✅ Fixed: Removed {duplicates} duplicates")
        else:
            print("\n✅ No duplicates found")
    
    # 5. データ範囲のチェック
    if 'y' in df.columns:
        y_stats = df['y'].describe()
        print(f"\n📊 Value statistics:")
        print(f"   Min: {y_stats['min']:.2f}")
        print(f"   Max: {y_stats['max']:.2f}")
        print(f"   Mean: {y_stats['mean']:.2f}")
        print(f"   Std: {y_stats['std']:.2f}")
        
        # 極端な外れ値の警告
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['y'] < Q1 - 3*IQR) | (df['y'] > Q3 + 3*IQR)]
        
        if len(outliers) > 0:
            print(f"   ⚠️  {len(outliers)} extreme outliers detected")
    
    # サマリー
    print("\n" + "=" * 60)
    if not issues:
        print("✅ Dataset is clean and ready to use!")
    else:
        print(f"❌ Found {len(issues)} issue(s)")
        if fix and fixes:
            print(f"✅ Applied {len(fixes)} fix(es)")
    print("=" * 60)
    
    return df if fix else None, issues, fixes

# 使用例
df = pd.read_csv('your_data.csv')

# 診断のみ
diagnose_and_fix_dataset(df, fix=False)

# 診断と修正
df_fixed, issues, fixes = diagnose_and_fix_dataset(df, fix=True)
if df_fixed is not None:
    df_fixed.to_csv('data_fixed.csv', index=False)
    print("\n💾 Fixed dataset saved to 'data_fixed.csv'")
```

---

### 問題: 予測ホライゾンが大きすぎる

#### 症状
```
ValidationWarning: Forecast horizon (h=365) is very large compared to dataset size
```

#### 解決策

**推奨ガイドライン**:
- 予測ホライゾン `h` は、1系列あたりのデータポイント数の10-20%以下が理想
- 長期予測が必要な場合は、より多くのデータを収集

**修正例**:
```python
import pandas as pd

df = pd.read_csv('data.csv')

# 系列ごとのデータポイント数を確認
points_per_series = df.groupby('unique_id').size()
min_points = points_per_series.min()
max_points = points_per_series.max()

print(f"Data points per series:")
print(f"  Min: {min_points}")
print(f"  Max: {max_points}")
print(f"  Mean: {points_per_series.mean():.0f}")

# 推奨予測ホライゾン
recommended_h = int(min_points * 0.15)  # 15%
print(f"\nRecommended h: {recommended_h}")

# 長期予測が必要な場合の対策
if recommended_h < 30:  # 例: 30日分の予測が必要
    print("\n⚠️  Need more historical data for long-term forecasting")
    print("Options:")
    print("1. Collect more historical data")
    print("2. Use hierarchical forecasting")
    print("3. Split into multiple shorter-horizon forecasts")
```

---

## メモリとパフォーマンスの問題

### 問題: OOM (Out of Memory) エラー

#### 症状
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
MemoryError: Unable to allocate array with shape...
```

#### 解決策

**解決法1**: バッチサイズの削減
```python
from ray import tune

# メモリ効率の良い設定
memory_efficient_config = {
    'batch_size': tune.choice([16, 32]),  # 小さいバッチ
    'input_size': tune.choice([7, 14]),   # 短いルックバック
    'hidden_size': tune.choice([128, 256])  # 小さいモデル
}

auto_model = create_auto_model(
    model_name="NHITS",
    h=7,
    dataset=df,
    config=memory_efficient_config,
    gpus=1
)
```

**解決法2**: GPUメモリのクリア
```python
import torch
import gc

def clear_gpu_memory():
    """GPUメモリをクリア"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU memory cleared")
        
        # 空きメモリの確認
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            cached = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            free = total - allocated
            print(f"GPU {i}: {free:.2f}GB free / {total:.2f}GB total")

# 使用前にメモリをクリア
clear_gpu_memory()

auto_model = create_auto_model(...)
```

**解決法3**: データのサブサンプリング
```python
import pandas as pd

def subsample_data(df, fraction=0.5, random_state=42):
    """データをサブサンプリング"""
    series_ids = df['unique_id'].unique()
    n_sample = int(len(series_ids) * fraction)
    
    sampled_ids = pd.Series(series_ids).sample(
        n=n_sample,
        random_state=random_state
    )
    
    df_sampled = df[df['unique_id'].isin(sampled_ids)]
    
    print(f"Subsampled: {len(df_sampled)} rows ({len(df)} original)")
    print(f"Series: {n_sample} / {len(series_ids)}")
    
    return df_sampled

# 大規模データの場合
if len(df) > 100000:
    df_train = subsample_data(df, fraction=0.5)
else:
    df_train = df
```

**解決法4**: 複数GPUの使用
```python
from auto_model_factory import OptimizationConfig

# マルチGPU設定
opt_config = OptimizationConfig(
    backend="ray",  # Rayは複数GPUに優れている
    num_samples=30,
    cpus=16,
    gpus=4,  # 複数GPU
    use_pruning=True
)

# メモリが4つのGPUに分散される
auto_model = create_auto_model(
    model_name="TFT",
    h=24,
    dataset=df,
    optimization_config=opt_config
)
```

---

### 問題: 学習が遅い

#### 症状
- 最適化に予想以上の時間がかかる
- GPUが100%使用されていない

#### 解決策

**診断スクリプト**:
```python
import time
import torch
from auto_model_factory import create_auto_model

def measure_optimization_speed(df, model_name, num_samples=5):
    """最適化速度を測定"""
    
    print(f"🔍 Measuring speed for {model_name}...")
    print(f"   Trials: {num_samples}")
    print(f"   GPU: {torch.cuda.is_available()}")
    
    start_time = time.time()
    
    auto_model = create_auto_model(
        model_name=model_name,
        h=7,
        dataset=df,
        num_samples=num_samples,
        gpus=1 if torch.cuda.is_available() else 0,
        verbose=False
    )
    
    elapsed = time.time() - start_time
    time_per_trial = elapsed / num_samples
    
    print(f"\n📊 Results:")
    print(f"   Total time: {elapsed:.1f}s")
    print(f"   Time per trial: {time_per_trial:.1f}s")
    print(f"   Estimated for 50 trials: {time_per_trial * 50 / 60:.1f} min")
    
    return time_per_trial

# 使用例
df = pd.read_csv('data.csv')
measure_optimization_speed(df, "NHITS", num_samples=5)
```

**高速化策**:

1. **試行回数の最適化**:
```python
from search_algorithm_selector import recommend_num_samples, ModelComplexity, DatasetSize

# データとモデルに応じた推奨値を使用
num_samples, explanation = recommend_num_samples(
    model_complexity=ModelComplexity.MODERATE,
    dataset_size=DatasetSize.MEDIUM,
    time_budget_hours=1.0  # 1時間以内
)

print(f"Recommended trials: {num_samples}")
```

2. **早期停止の活用**:
```python
from auto_model_factory import OptimizationConfig

opt_config = OptimizationConfig(
    backend="optuna",
    num_samples=50,
    use_pruning=True,  # 見込みのない試行を早期停止
    time_budget_hours=2.0  # 時間制限
)
```

3. **並列実行**:
```python
# Ray Tuneで並列実行
opt_config = OptimizationConfig(
    backend="ray",
    num_samples=40,
    cpus=16,  # 多くのCPU
    gpus=2    # 複数GPU
)

# Optunaでも並列可能（実験的）
import optuna
optuna.create_study(n_jobs=4)  # 4並列
```

4. **シンプルなモデルから開始**:
```python
# まず軽量モデルで設定を確認
quick_model = create_auto_model(
    model_name="MLP",  # 最もシンプル
    h=7,
    dataset=df,
    num_samples=10
)

# 良好なら本格的なモデル
final_model = create_auto_model(
    model_name="TFT",
    h=7,
    dataset=df,
    num_samples=50
)
```

---

## 最適化の問題

### 問題: 最適化が収束しない

#### 症状
- 試行を重ねてもメトリクスが改善しない
- 最良スコアが変わらない
- 検証エラーが高止まり

#### 解決策

**診断スクリプト**:
```python
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

def analyze_optimization_history(experiment_name):
    """最適化履歴を分析"""
    
    # MLflowから履歴を取得
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Experiment '{experiment_name}' not found")
        return
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time ASC"]
    )
    
    if len(runs) == 0:
        print("❌ No runs found")
        return
    
    print(f"📊 Optimization Analysis for '{experiment_name}'")
    print(f"   Total trials: {len(runs)}")
    
    # メトリクスの抽出（損失値など）
    metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
    
    if len(metric_cols) == 0:
        print("⚠️  No metrics found")
        return
    
    # 最初のメトリクスを使用
    metric_col = metric_cols[0]
    metric_name = metric_col.replace('metrics.', '')
    
    values = runs[metric_col].dropna()
    
    print(f"\n📈 Metric: {metric_name}")
    print(f"   Best: {values.min():.4f}")
    print(f"   Worst: {values.max():.4f}")
    print(f"   Mean: {values.mean():.4f}")
    print(f"   Std: {values.std():.4f}")
    
    # 収束の判定
    # 最初の10試行と最後の10試行を比較
    if len(values) >= 20:
        first_10 = values.iloc[:10].mean()
        last_10 = values.iloc[-10:].mean()
        improvement = (first_10 - last_10) / first_10 * 100
        
        print(f"\n🎯 Convergence:")
        print(f"   First 10 trials avg: {first_10:.4f}")
        print(f"   Last 10 trials avg: {last_10:.4f}")
        print(f"   Improvement: {improvement:.1f}%")
        
        if improvement < 5:
            print("\n⚠️  Warning: Poor convergence detected")
            print("Possible causes:")
            print("1. Search space too large")
            print("2. Learning rate issues")
            print("3. Model too complex for data")
            print("4. Need more trials")
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(values.values)
    plt.axhline(values.min(), color='r', linestyle='--', label='Best')
    plt.xlabel('Trial')
    plt.ylabel(metric_name)
    plt.title(f'Optimization History: {experiment_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{experiment_name}_history.png')
    print(f"\n💾 Plot saved to '{experiment_name}_history.png'")

# 使用例
analyze_optimization_history("my_experiment")
```

**解決策**:

1. **探索空間を狭める**:
```python
from ray import tune

# 広すぎる探索空間（収束しにくい）
wide_config = {
    'max_steps': tune.choice([500, 1000, 2000, 3000, 5000]),
    'learning_rate': tune.loguniform(1e-6, 1e-1),
    'batch_size': tune.choice([16, 32, 64, 128, 256, 512])
}

# 焦点を絞った探索空間（収束しやすい）
focused_config = {
    'max_steps': tune.choice([1000, 2000]),  # 狭い範囲
    'learning_rate': tune.loguniform(1e-4, 1e-2),  # 現実的な範囲
    'batch_size': tune.choice([64, 128])  # 2つの選択肢
}
```

2. **試行回数を増やす**:
```python
# 不十分
auto_model = create_auto_model(..., num_samples=10)

# より良い
auto_model = create_auto_model(..., num_samples=50)

# 徹底的
auto_model = create_auto_model(..., num_samples=100)
```

3. **異なるアルゴリズムを試す**:
```python
from search_algorithm_selector import SearchAlgorithmSelector, ModelComplexity, DatasetSize

selector = SearchAlgorithmSelector(backend="optuna")

# 異なる戦略を試す
strategies = []

# TPE（デフォルト）
strategy_tpe = selector.select_algorithm(
    model_complexity=ModelComplexity.MODERATE,
    dataset_size=DatasetSize.MEDIUM,
    num_samples=50
)
strategies.append(("TPE", strategy_tpe))

# CMA-ES（連続パラメータに強い）
strategy_cmaes = selector.select_algorithm(
    model_complexity=ModelComplexity.SIMPLE,
    dataset_size=DatasetSize.SMALL,
    num_samples=50
)
strategies.append(("CMA-ES", strategy_cmaes))

for name, strategy in strategies:
    print(f"Testing {name}: {strategy.algorithm_name}")
```

4. **デフォルト設定から開始**:
```python
# カスタム設定なしで試す（デフォルトは調整済み）
auto_model = create_auto_model(
    model_name="NHITS",
    h=7,
    dataset=df,
    config=None,  # デフォルト使用
    num_samples=50
)
```

---

### 問題: すべての試行が失敗する

#### 症状
```
OptimizationError: All trials failed
RuntimeError: Trial execution failed
```

#### 解決策

**デバッグモードで実行**:
```python
import logging
from auto_model_factory import create_auto_model

# 詳細ログを有効化
logging.basicConfig(level=logging.DEBUG)

try:
    auto_model = create_auto_model(
        model_name="NHITS",
        h=7,
        dataset=df,
        num_samples=1,  # まず1試行だけ
        verbose=True
    )
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
```

**一般的な原因と解決**:

1. **データの問題**:
```python
from validation import DataValidator

# データを検証
validator = DataValidator()
result = validator.validate_dataset(df)

if not result.is_valid:
    print("Data issues:")
    for error in result.errors:
        print(f"  - {error}")
```

2. **設定の問題**:
```python
from validation import ConfigValidator

# 設定を検証
validator = ConfigValidator(strict_mode=True)
result = validator.validate_backend_config(
    backend="optuna",
    config=my_config,
    num_samples=50,
    cpus=4,
    gpus=1
)

if not result.is_valid:
    print("Config issues:")
    for error in result.errors:
        print(f"  - {error}")
    
    if result.corrected_config:
        print("Suggested corrections:")
        print(result.corrected_config)
```

3. **リソースの問題**:
```python
# シンプルな設定で試す
minimal_config = {
    'max_steps': 100,  # 非常に少ない
    'batch_size': 32,
    'learning_rate': 0.001
}

auto_model = create_auto_model(
    model_name="MLP",  # 最もシンプルなモデル
    h=7,
    dataset=df.head(1000),  # データも削減
    config=minimal_config,
    num_samples=1,
    gpus=0  # CPUのみ
)
```

---

## MLflowの問題

### 問題: MLflowサーバーに接続できない

#### 症状
```
MlflowException: Connection refused
RequestException: Connection error
```

#### 解決策

**診断と解決**:
```python
from validation import ConfigValidator
import mlflow
import requests

def diagnose_mlflow_connection(tracking_uri="http://localhost:5000"):
    """MLflow接続を診断"""
    
    print(f"🔍 Diagnosing MLflow connection: {tracking_uri}\n")
    
    # 1. サーバーの応答確認
    try:
        response = requests.get(tracking_uri, timeout=5)
        print(f"✅ Server responded: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Connection refused")
        print("Solutions:")
        print("1. Start MLflow server: mlflow ui --host 0.0.0.0 --port 5000")
        print("2. Check if port is already in use: netstat -an | grep 5000")
        print("3. Try different port: mlflow ui --port 5001")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timeout")
        print("Server may be overloaded or network issues")
        return False
    
    # 2. MLflow APIの確認
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiments = mlflow.search_experiments()
        print(f"✅ MLflow API works: {len(experiments)} experiments found")
        return True
    except Exception as e:
        print(f"❌ MLflow API error: {e}")
        return False

# 診断実行
is_connected = diagnose_mlflow_connection()

if not is_connected:
    print("\n💡 Workaround: Disable MLflow")
    print("auto_model = create_auto_model(..., use_mlflow=False)")
```

**MLflowサーバーの起動**:
```bash
# 基本的な起動
mlflow ui

# カスタムポート
mlflow ui --port 5001

# 外部からアクセス可能にする
mlflow ui --host 0.0.0.0 --port 5000

# バックエンドストアを指定
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

### 問題: MLflowの実験が重複する

#### 症状
- 同じ名前の実験が複数作成される
- 実験の管理が混乱する

#### 解決策

```python
import mlflow

def get_or_create_experiment(experiment_name):
    """実験を取得または作成（重複を避ける）"""
    
    # 既存の実験を検索
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        # 存在しない場合は作成
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created new experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"✅ Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

# 使用例
experiment_id = get_or_create_experiment("my_forecasting_project")

# これで安全に実験を実行
with mlflow.start_run():
    auto_model = create_auto_model(
        model_name="NHITS",
        h=7,
        dataset=df,
        use_mlflow=True,
        verbose=True
    )
```

---

## GPU関連の問題

### 問題: GPUメモリリーク

#### 症状
- 実行を重ねるとGPUメモリが解放されない
- 2回目以降の実行でOOMエラー

#### 解決策

```python
import torch
import gc

def reset_gpu_state():
    """GPUの状態を完全にリセット"""
    
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    
    print("🔄 Resetting GPU state...")
    
    # すべてのGPUキャッシュをクリア
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    # Pythonのガベージコレクション
    gc.collect()
    
    print("✅ GPU state reset complete")
    
    # メモリ状態を表示
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# 実験の間に実行
reset_gpu_state()

auto_model = create_auto_model(...)

reset_gpu_state()  # 実験後もリセット
```

**プロセス分離**:
```python
import subprocess

def run_optimization_in_subprocess(script_path):
    """サブプロセスで最適化を実行（メモリ完全クリーン）"""
    
    result = subprocess.run(
        ['python', script_path],
        capture_output=True,
        text=True
    )
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)
    
    return result.returncode == 0

# メインスクリプト
# optimization_script.py を作成して使用
success = run_optimization_in_subprocess('optimization_script.py')
```

---

## エラーメッセージ別ガイド

### "ValueError: Unknown model name"

**原因**: サポートされていないモデル名

**解決法**:
```python
from model_characteristics import MODEL_CATALOG

# サポートされるモデルを確認
supported_models = list(MODEL_CATALOG.keys())
print("Supported models:", supported_models)

# 正しいモデル名を使用
auto_model = create_auto_model(
    model_name="NHITS",  # 正しい名前
    # model_name="nhits",  # ❌ 小文字は不可
    # model_name="N-HITS",  # ❌ ハイフンは不可
    h=7,
    dataset=df
)
```

---

### "RuntimeError: Expected tensor for argument"

**原因**: データ型の不一致

**解決法**:
```python
import pandas as pd

# データ型を確認
print(df.dtypes)

# 正しい型に変換
df['unique_id'] = df['unique_id'].astype(str)
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# NaNを処理
df = df.dropna()

# 再試行
auto_model = create_auto_model(...)
```

---

### "KeyError: 'unique_id'"

**原因**: 必須カラムが存在しない

**解決法**:
```python
# カラム名を確認
print("Columns:", df.columns.tolist())

# カラム名をリネーム
df = df.rename(columns={
    'series_id': 'unique_id',  # 例
    'date': 'ds',
    'value': 'y'
})

# 再試行
auto_model = create_auto_model(...)
```

---

## デバッグ手法

### ステップバイステップデバッグ

```python
import pandas as pd
from auto_model_factory import create_auto_model
import logging

# 1. ロギングを有効化
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 2. データを確認
print("Step 1: Loading data")
df = pd.read_csv('data.csv')
print(f"  Loaded: {len(df)} rows")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Data types:\n{df.dtypes}")

# 3. データをクリーン
print("\nStep 2: Cleaning data")
df['ds'] = pd.to_datetime(df['ds'])
df = df.dropna()
print(f"  After cleaning: {len(df)} rows")

# 4. 小規模テスト
print("\nStep 3: Small-scale test")
df_test = df.head(1000)  # 最初の1000行のみ
try:
    auto_model = create_auto_model(
        model_name="MLP",  # シンプルなモデル
        h=7,
        dataset=df_test,
        num_samples=1,  # 1試行のみ
        verbose=True,
        gpus=0  # CPUのみ
    )
    print("  ✅ Small test passed")
except Exception as e:
    print(f"  ❌ Small test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. 徐々にスケールアップ
print("\nStep 4: Scaling up")
try:
    auto_model = create_auto_model(
        model_name="NHITS",
        h=7,
        dataset=df,  # 全データ
        num_samples=10,  # より多くの試行
        verbose=True,
        gpus=1  # GPU使用
    )
    print("  ✅ Full run passed")
except Exception as e:
    print(f"  ❌ Full run failed: {e}")
    traceback.print_exc()
```

---

### パフォーマンスプロファイリング

```python
import cProfile
import pstats
import io

def profile_optimization(df):
    """最適化をプロファイリング"""
    
    pr = cProfile.Profile()
    pr.enable()
    
    # 最適化実行
    auto_model = create_auto_model(
        model_name="NHITS",
        h=7,
        dataset=df,
        num_samples=5,
        verbose=False
    )
    
    pr.disable()
    
    # 結果を表示
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # 上位20件
    
    print(s.getvalue())

# 使用
profile_optimization(df)
```

---

## FAQ

### Q1: 最初に何を確認すべきですか？

**A**: まず`diagnostic_check.py`スクリプトを実行して、システム環境を確認してください。

---

### Q2: エラーメッセージが理解できません

**A**: 
1. エラーメッセージ全体をコピー
2. このドキュメントの「エラーメッセージ別ガイド」セクションを検索
3. 見つからない場合は、スタックトレースの最後の行を確認
4. それでも解決しない場合は、GitHubでIssueを作成

---

### Q3: 最適化に時間がかかりすぎます

**A**: 
1. `num_samples`を減らす（10-20に）
2. `use_pruning=True`を設定
3. `time_budget_hours`で時間制限を設定
4. シンプルなモデル（MLP、DLinear）を試す

---

### Q4: 精度が期待より低い

**A**:
1. より多くのデータを収集
2. `num_samples`を増やす（50-100に）
3. 異なるモデルを試す
4. `input_size`を調整
5. カスタム損失関数を使用

---

### Q5: GPU が認識されません

**A**:
1. `nvidia-smi`でドライバーを確認
2. PyTorchのCUDA版を再インストール
3. `torch.cuda.is_available()`で確認
4. 最後の手段: `gpus=0`でCPU-onlyモードを使用

---

## サポートを受ける前に

問題を報告する前に、以下の情報を収集してください：

```python
# bug_report.py - バグレポート情報を収集
import sys
import torch
import pandas as pd

def generate_bug_report():
    """バグレポート用の情報を収集"""
    
    report = []
    report.append("=" * 80)
    report.append("BUG REPORT")
    report.append("=" * 80)
    
    # システム情報
    report.append("\n## System Information")
    report.append(f"Python version: {sys.version}")
    report.append(f"PyTorch version: {torch.__version__}")
    report.append(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        report.append(f"CUDA version: {torch.version.cuda}")
        report.append(f"GPU count: {torch.cuda.device_count()}")
    
    # パッケージバージョン
    report.append("\n## Package Versions")
    packages = ['neuralforecast', 'optuna', 'ray', 'mlflow', 'pandas', 'numpy']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            report.append(f"{pkg}: {version}")
        except ImportError:
            report.append(f"{pkg}: NOT INSTALLED")
    
    # 問題の説明（手動で編集）
    report.append("\n## Problem Description")
    report.append("Please describe your problem here:")
    report.append("")
    
    # エラーメッセージ（手動で編集）
    report.append("\n## Error Message")
    report.append("Paste full error message and stack trace here:")
    report.append("")
    
    # 再現コード（手動で編集）
    report.append("\n## Minimal Reproducible Code")
    report.append("```python")
    report.append("# Paste your code here")
    report.append("```")
    
    report_text = "\n".join(report)
    
    # ファイルに保存
    with open('bug_report.txt', 'w') as f:
        f.write(report_text)
    
    print("📝 Bug report template saved to 'bug_report.txt'")
    print("Please fill in the problem description, error message, and code")
    print("Then submit to GitHub Issues")
    
    return report_text

generate_bug_report()
```

---

## 🆘 最終手段

すべての解決策を試しても問題が解決しない場合：

1. **GitHubでIssueを作成**: バグレポートを含めて
2. **コミュニティフォーラムで質問**: Discussions セクションで
3. **ドキュメントを再確認**: [README](./README.md)、[TUTORIAL](./TUTORIAL.md)、[API_REFERENCE](./API_REFERENCE.md)

**問題解決の旅、応援しています！🚀**
