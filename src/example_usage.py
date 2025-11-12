#!/usr/bin/env python3
"""
自動モデルファクトリーの使用例
このスクリプトは、作成した3つのモジュールの使用方法を示します。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_series: int = 5, n_obs: int = 365) -> pd.DataFrame:
    """
    サンプルデータの生成
    
    Args:
        n_series: 時系列の数
        n_obs: 各時系列の観測数
        
    Returns:
        pd.DataFrame: サンプルデータ
    """
    logger.info(f"Generating sample data: {n_series} series × {n_obs} observations")
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for series_id in range(n_series):
        # トレンド + 季節性 + ノイズ
        trend = np.linspace(0, 10, n_obs)
        seasonality = 5 * np.sin(np.linspace(0, 4 * np.pi, n_obs))
        noise = np.random.normal(0, 1, n_obs)
        y = 50 + trend + seasonality + noise
        
        for i in range(n_obs):
            data.append({
                'unique_id': f'series_{series_id}',
                'ds': start_date + timedelta(days=i),
                'y': y[i]
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated data shape: {df.shape}")
    return df


def example_1_basic_usage():
    """
    例1: 基本的な使用方法
    """
    print("\n" + "=" * 80)
    print("例1: 基本的な使用方法")
    print("=" * 80)
    
    # インポート
    import sys
    sys.path.append('/home/claude/src')
    from auto_model_factory import create_auto_model
    
    # サンプルデータ生成
    df = generate_sample_data(n_series=3, n_obs=200)
    
    # 自動最適化実行
    logger.info("Starting basic auto-optimization...")
    
    try:
        auto_model = create_auto_model(
            model_name="NHITS",
            h=7,  # 7日先予測
            dataset=df,
            backend="optuna",
            num_samples=5,  # デモ用に少なめ
            cpus=2,
            gpus=0,
            use_mlflow=False,  # デモではMLflowを無効化
            verbose=True
        )
        
        logger.info("✓ Optimization completed successfully!")
        
        # 予測
        predictions = auto_model.predict(dataset=df)
        logger.info(f"Predictions shape: {predictions.shape}")
        print("\nPredictions (first 10 rows):")
        print(predictions.head(10))
        
    except Exception as e:
        logger.error(f"Error in basic usage example: {e}")
        import traceback
        traceback.print_exc()


def example_2_with_validation():
    """
    例2: 検証機能の使用
    """
    print("\n" + "=" * 80)
    print("例2: 検証機能の使用")
    print("=" * 80)
    
    import sys
    sys.path.append('/home/claude/src')
    from validation import (
        ConfigValidator, 
        DataValidator,
        validate_all,
        print_validation_results
    )
    
    # サンプルデータ
    df = generate_sample_data(n_series=3, n_obs=200)
    
    # 検証実行
    logger.info("Running validations...")
    
    config = {
        'max_steps': 1000,
        'learning_rate': 0.001,
        'batch_size': 64
    }
    
    validation_results = validate_all(
        backend="optuna",
        config=config,
        num_samples=20,
        cpus=4,
        gpus=0,
        model_class_name="NHITS",
        dataset=df,
        h=7,
        strict_mode=False
    )
    
    # 結果表示
    all_valid = print_validation_results(validation_results)
    
    if all_valid:
        logger.info("✓ All validations passed!")
    else:
        logger.warning("⚠ Some validations failed")


def example_3_algorithm_selection():
    """
    例3: 探索アルゴリズムの選択
    """
    print("\n" + "=" * 80)
    print("例3: 探索アルゴリズムの選択")
    print("=" * 80)
    
    import sys
    sys.path.append('/home/claude/src')
    from search_algorithm_selector import (
        SearchAlgorithmSelector,
        ModelComplexity,
        DatasetSize,
        SearchComplexity,
        recommend_num_samples
    )
    
    # アルゴリズム選択
    logger.info("Selecting search algorithm...")
    
    selector = SearchAlgorithmSelector(backend="optuna")
    
    config = {
        'max_steps': [500, 1000, 2000],
        'learning_rate': (0.0001, 0.01),
        'batch_size': [32, 64, 128]
    }
    
    strategy = selector.select_algorithm(
        model_complexity=ModelComplexity.MODERATE,
        dataset_size=DatasetSize.MEDIUM,
        num_samples=50,
        config=config,
        use_pruning=True,
        random_seed=42
    )
    
    print(f"\n選択された戦略:")
    print(f"  Algorithm: {strategy.algorithm_name}")
    print(f"  Pruner: {strategy.pruner_name}")
    print(f"  Description: {strategy.description}")
    print(f"  Sampler kwargs: {strategy.sampler_kwargs}")
    
    # 試行回数の推奨
    logger.info("\nRecommending number of samples...")
    
    num_samples, explanation = recommend_num_samples(
        model_complexity=ModelComplexity.COMPLEX,
        dataset_size=DatasetSize.LARGE,
        search_complexity=SearchComplexity.HIGH,
        time_budget_hours=2.0
    )
    
    print(f"\n推奨試行回数:")
    print(f"  Samples: {num_samples}")
    print(f"  Explanation: {explanation}")


def example_4_advanced_factory():
    """
    例4: 高度なファクトリーの使用
    """
    print("\n" + "=" * 80)
    print("例4: 高度なファクトリーの使用")
    print("=" * 80)
    
    import sys
    sys.path.append('/home/claude/src')
    from auto_model_factory import (
        AutoModelFactory, 
        OptimizationConfig,
        MODEL_CATALOG
    )
    
    # サンプルデータ
    df = generate_sample_data(n_series=5, n_obs=300)
    
    # モデルカタログの確認
    logger.info("Available models in catalog:")
    for model_name, char in MODEL_CATALOG.items():
        print(f"  - {model_name}: {char.complexity.value}, "
              f"memory={char.memory_footprint_mb}MB, "
              f"time={char.typical_training_time_minutes}min")
    
    # 最適化設定
    opt_config = OptimizationConfig(
        backend="optuna",
        num_samples=None,  # 自動推奨
        cpus=4,
        gpus=0,
        use_mlflow=False,
        use_pruning=True,
        time_budget_hours=0.5,  # 30分
        verbose=True,
        strict_validation=False
    )
    
    # ファクトリー作成
    logger.info("\nCreating AutoModelFactory...")
    
    try:
        factory = AutoModelFactory(
            model_name="NHITS",
            h=7,
            optimization_config=opt_config
        )
        
        # 最適化実行
        logger.info("Starting optimization with factory...")
        
        auto_model = factory.create_auto_model(
            dataset=df,
            config=None  # デフォルト設定を使用
        )
        
        logger.info("✓ Factory optimization completed!")
        
        # 最適化履歴の確認
        history = factory.get_optimization_summary()
        if not history.empty:
            print("\n最適化履歴:")
            print(history)
        
    except Exception as e:
        logger.error(f"Error in advanced factory example: {e}")
        import traceback
        traceback.print_exc()


def example_5_custom_config():
    """
    例5: カスタム設定の使用
    """
    print("\n" + "=" * 80)
    print("例5: カスタム設定の使用")
    print("=" * 80)
    
    import sys
    sys.path.append('/home/claude/src')
    
    # Ray Tuneのインポート（必要な場合のみ）
    try:
        from ray import tune
        
        logger.info("Creating custom configuration with Ray Tune...")
        
        # カスタムハイパーパラメータ探索空間
        custom_config = {
            'max_steps': tune.choice([500, 1000, 1500]),
            'learning_rate': tune.loguniform(1e-4, 1e-2),
            'batch_size': tune.choice([32, 64, 128]),
            'input_size': tune.choice([7, 14, 21]),
            'n_blocks': tune.choice([[1, 1, 1], [2, 2, 2]]),
        }
        
        print("\nカスタム設定:")
        for key, value in custom_config.items():
            print(f"  {key}: {value}")
        
        # データ生成
        df = generate_sample_data(n_series=3, n_obs=200)
        
        # カスタム設定で実行
        from auto_model_factory import create_auto_model
        
        logger.info("Running optimization with custom config...")
        
        auto_model = create_auto_model(
            model_name="NHITS",
            h=7,
            dataset=df,
            config=custom_config,
            backend="ray",  # Rayバックエンドを使用
            num_samples=5,
            cpus=2,
            gpus=0,
            use_mlflow=False,
            verbose=True
        )
        
        logger.info("✓ Custom config optimization completed!")
        
    except ImportError:
        logger.warning("Ray Tune not available. Skipping custom config example.")
        logger.info("Install with: pip install 'ray[tune]'")
    except Exception as e:
        logger.error(f"Error in custom config example: {e}")
        import traceback
        traceback.print_exc()


def example_6_data_validation():
    """
    例6: データ検証の詳細
    """
    print("\n" + "=" * 80)
    print("例6: データ検証の詳細")
    print("=" * 80)
    
    import sys
    sys.path.append('/home/claude/src')
    from validation import DataValidator
    
    # 正常なデータ
    logger.info("Testing with valid data...")
    valid_df = generate_sample_data(n_series=3, n_obs=200)
    
    validator = DataValidator()
    result = validator.validate_dataset(valid_df)
    
    print("\n正常なデータの検証結果:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")
    
    # 異常なデータ（必須カラム欠損）
    logger.info("\nTesting with invalid data (missing columns)...")
    invalid_df = valid_df.drop(columns=['y'])
    
    result = validator.validate_dataset(invalid_df)
    
    print("\n異常なデータの検証結果:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # ホライゾン検証
    logger.info("\nTesting horizon validation...")
    result = validator.validate_horizon(
        h=50,
        dataset_length=len(valid_df),
        input_size=14
    )
    
    print("\nホライゾン検証結果:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Warnings: {result.warnings}")


def main():
    """
    メイン関数 - 全ての例を実行
    """
    print("\n" + "=" * 80)
    print("自動モデルファクトリー - 使用例デモンストレーション")
    print("=" * 80)
    
    examples = [
        ("基本的な使用方法", example_1_basic_usage),
        ("検証機能の使用", example_2_with_validation),
        ("探索アルゴリズムの選択", example_3_algorithm_selection),
        ("高度なファクトリーの使用", example_4_advanced_factory),
        ("カスタム設定の使用", example_5_custom_config),
        ("データ検証の詳細", example_6_data_validation),
    ]
    
    print("\n利用可能な例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n全ての例を実行しますか？ (y/n, または実行したい例の番号): ", end="")
    
    # デモモードでは自動的に例2, 3, 6のみを実行
    # （実際の実行は時間がかかるため）
    demo_examples = [2, 3, 6]
    
    logger.info(f"\nデモモード: 例 {demo_examples} を実行します")
    
    for idx in demo_examples:
        if 1 <= idx <= len(examples):
            name, func = examples[idx - 1]
            try:
                func()
            except Exception as e:
                logger.error(f"Error in example {idx} ({name}): {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("デモンストレーション完了")
    print("=" * 80)
    print("\n注意: 例1, 4, 5 は実際の最適化を実行するため時間がかかります。")
    print("実際に実行する場合は、スクリプトを直接編集してください。")


if __name__ == "__main__":
    main()
