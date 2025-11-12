"""
設定ファイル使用例

このスクリプトは、config_loader.pyを使って設定ファイルを
読み込み、活用する方法を示します。
"""

from config_loader import (
    ConfigLoader,
    load_default_config,
    load_model_characteristics,
    get_model_config
)
from mlflow_integration import MLflowTracker
from search_algorithm_selector import SearchAlgorithmSelector
import logging


def example_1_basic_usage():
    """
    例1: 基本的な使用方法
    """
    print("=" * 60)
    print("例1: 基本的な設定ファイルの読み込み")
    print("=" * 60)
    
    # 設定を読み込む
    config = load_default_config()
    
    # ドット記法でアクセス
    print(f"MLflow tracking URI: {config.mlflow.tracking_uri}")
    print(f"Default experiment: {config.mlflow.default_experiment_name}")
    print(f"Optuna sampler: {config.optuna.sampler.default}")
    print(f"Max epochs: {config.training.max_epochs}")
    print()


def example_2_model_characteristics():
    """
    例2: モデル特性の読み込み
    """
    print("=" * 60)
    print("例2: モデル特性の読み込みと活用")
    print("=" * 60)
    
    # モデル特性を読み込む
    model_chars = load_model_characteristics()
    
    # 全モデル名を取得
    all_models = list(model_chars.models.keys())
    print(f"利用可能なモデル数: {len(all_models)}")
    print(f"モデル例: {all_models[:5]}")
    print()
    
    # 特定のモデルの特性を取得
    tft_config = get_model_config("AutoTFT")
    print("AutoTFTの特性:")
    print(f"  - 探索空間サイズ: {tft_config.typical_search_space_size}")
    print(f"  - ドロップアウト対応: {tft_config.dropout_supported}")
    print(f"  - デフォルトスケーラー: {tft_config.default_scaler}")
    print(f"  - 外生変数サポート: {tft_config.exogenous_support}")
    print(f"  - 備考: {tft_config.notes}")
    print()


def example_3_mlflow_integration():
    """
    例3: MLflow統合での使用
    """
    print("=" * 60)
    print("例3: MLflow統合での設定活用")
    print("=" * 60)
    
    # 設定を読み込む
    config = load_default_config()
    mlflow_config = config.mlflow
    
    # MLflowTrackerを初期化
    tracker = MLflowTracker(
        tracking_uri=mlflow_config.tracking_uri,
        experiment_name=mlflow_config.default_experiment_name,
        tags=dict(mlflow_config.experiment_tags)
    )
    
    print(f"MLflowTracker initialized")
    print(f"  Tracking URI: {mlflow_config.tracking_uri}")
    print(f"  Experiment: {mlflow_config.default_experiment_name}")
    print()


def example_4_search_algorithm_selection():
    """
    例4: 検索アルゴリズム選択での使用
    """
    print("=" * 60)
    print("例4: 検索アルゴリズム自動選択")
    print("=" * 60)
    
    # 設定を読み込む
    config = load_default_config()
    
    # SearchAlgorithmSelectorを初期化
    selector = SearchAlgorithmSelector(
        backend=config.search_algorithm_selector.backend
    )
    
    # モデル特性を取得
    tft_config = get_model_config("AutoTFT")
    
    # 検索アルゴリズムを選択（仮想的な設定）
    dummy_config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "hidden_size": 256,
        "dropout": 0.1,
        "num_layers": 3
    }
    
    strategy = selector.select_algorithm(
        model_complexity="complex",
        dataset_size=50000,  # 中規模データセット
        num_samples=config.optuna.optimization.n_trials,
        config=dummy_config,
        use_pruning=config.optuna.pruner.enabled
    )
    
    print(f"選択されたアルゴリズム: {strategy.algorithm_name}")
    print(f"プルーナー: {strategy.pruner_name}")
    print(f"説明: {strategy.description}")
    print()


def example_5_environment_override():
    """
    例5: 環境変数によるオーバーライド
    """
    print("=" * 60)
    print("例5: 環境変数によるオーバーライド")
    print("=" * 60)
    
    import os
    
    # 環境変数を設定（デモ用）
    os.environ['NEURALFORECAST_MLFLOW_TRACKING_URI'] = 'http://mlflow-server:5000'
    os.environ['NEURALFORECAST_OPTUNA_OPTIMIZATION_N_TRIALS'] = '100'
    
    # 設定を再読み込み
    config = load_default_config()
    
    print(f"MLflow tracking URI (環境変数で上書き): {config.mlflow.tracking_uri}")
    print(f"Optuna試行回数 (環境変数で上書き): {config.optuna.optimization.n_trials}")
    
    # クリーンアップ
    del os.environ['NEURALFORECAST_MLFLOW_TRACKING_URI']
    del os.environ['NEURALFORECAST_OPTUNA_OPTIMIZATION_N_TRIALS']
    print()


def example_6_custom_override():
    """
    例6: カスタム設定のオーバーライド
    """
    print("=" * 60)
    print("例6: カスタム設定のオーバーライド")
    print("=" * 60)
    
    # カスタム設定
    custom_config = {
        'mlflow': {
            'tracking_uri': 'databricks',
            'default_experiment_name': 'production_forecasting'
        },
        'optuna': {
            'optimization': {
                'n_trials': 200,
                'n_jobs': 8
            }
        }
    }
    
    # 設定をマージ
    loader = ConfigLoader()
    config = loader.load_config(override_config=custom_config)
    
    print("カスタム設定を適用:")
    print(f"  MLflow tracking URI: {config.mlflow.tracking_uri}")
    print(f"  Experiment name: {config.mlflow.default_experiment_name}")
    print(f"  Optuna試行回数: {config.optuna.optimization.n_trials}")
    print(f"  並列ジョブ数: {config.optuna.optimization.n_jobs}")
    print()


def example_7_model_complexity_classification():
    """
    例7: モデル複雑度による分類
    """
    print("=" * 60)
    print("例7: モデル複雑度による分類")
    print("=" * 60)
    
    # モデル特性を読み込む
    model_chars = load_model_characteristics()
    
    # 複雑度別のモデルリスト
    complexity = model_chars.model_complexity
    
    print("モデル複雑度分類:")
    print(f"\nSIMPLE ({len(complexity.simple)}モデル):")
    for model in complexity.simple:
        print(f"  - {model}")
    
    print(f"\nMODERATE ({len(complexity.moderate)}モデル):")
    for model in complexity.moderate[:5]:  # 最初の5つのみ表示
        print(f"  - {model}")
    print(f"  ... (他{len(complexity.moderate) - 5}モデル)")
    
    print(f"\nCOMPLEX ({len(complexity.complex)}モデル):")
    for model in complexity.complex[:5]:  # 最初の5つのみ表示
        print(f"  - {model}")
    print(f"  ... (他{len(complexity.complex) - 5}モデル)")
    print()


def example_8_save_custom_config():
    """
    例8: カスタム設定の保存
    """
    print("=" * 60)
    print("例8: カスタム設定の保存")
    print("=" * 60)
    
    # 設定を読み込み
    loader = ConfigLoader()
    config = loader.load_config()
    
    # 設定を変更
    config.mlflow.tracking_uri = "http://production-mlflow:5000"
    config.optuna.optimization.n_trials = 500
    config.training.max_epochs = 200
    
    # 新しい設定として保存
    output_path = "configs/custom_production.yaml"
    loader.save_config(config, output_path)
    
    print(f"カスタム設定を保存しました: {output_path}")
    print()


def main():
    """全ての例を実行"""
    
    print("\n")
    print("=" * 60)
    print("NeuralForecast AutoModels - 設定ファイル使用例")
    print("=" * 60)
    print("\n")
    
    try:
        example_1_basic_usage()
        example_2_model_characteristics()
        example_3_mlflow_integration()
        example_4_search_algorithm_selection()
        example_5_environment_override()
        example_6_custom_override()
        example_7_model_complexity_classification()
        # example_8_save_custom_config()  # 実際にファイルを作成するのでコメントアウト
        
        print("=" * 60)
        print("全ての例が正常に実行されました")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # ログレベルを設定（オプション）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
