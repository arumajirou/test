# src/auto_model_factory.py
"""
自動モデルファクトリー
NeuralForecastモデルの自動ハイパーパラメータ最適化を
統合的に管理するメインモジュール
"""
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import json

import pandas as pd
import numpy as np

# プロジェクト内モジュール
from .validation import (
    ConfigValidator, 
    DataValidator, 
    validate_all,
    print_validation_results
)
from .search_algorithm_selector import (
    SearchAlgorithmSelector,
    ModelComplexity,
    DatasetSize,
    SearchComplexity,
    recommend_num_samples
)

logger = logging.getLogger(__name__)


@dataclass
class ModelCharacteristics:
    """モデル特性の定義"""
    name: str
    complexity: ModelComplexity
    recommended_input_size_range: tuple = (7, 24)
    supports_exogenous: bool = True
    supports_static: bool = False
    typical_training_time_minutes: float = 10.0
    memory_footprint_mb: float = 500.0
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"ModelCharacteristics(name={self.name}, complexity={self.complexity.value})"


# モデル特性カタログ
MODEL_CATALOG = {
    # シンプルなモデル
    "MLP": ModelCharacteristics(
        name="MLP",
        complexity=ModelComplexity.SIMPLE,
        recommended_input_size_range=(7, 14),
        typical_training_time_minutes=5.0,
        memory_footprint_mb=300.0
    ),
    "NHITS": ModelCharacteristics(
        name="NHITS",
        complexity=ModelComplexity.MODERATE,
        recommended_input_size_range=(14, 28),
        typical_training_time_minutes=10.0,
        memory_footprint_mb=800.0
    ),
    "NBEATS": ModelCharacteristics(
        name="NBEATS",
        complexity=ModelComplexity.MODERATE,
        recommended_input_size_range=(14, 28),
        typical_training_time_minutes=10.0,
        memory_footprint_mb=800.0
    ),
    # 中程度の複雑さ
    "DLinear": ModelCharacteristics(
        name="DLinear",
        complexity=ModelComplexity.MODERATE,
        recommended_input_size_range=(24, 96),
        typical_training_time_minutes=8.0,
        memory_footprint_mb=400.0
    ),
    "TSMixer": ModelCharacteristics(
        name="TSMixer",
        complexity=ModelComplexity.MODERATE,
        recommended_input_size_range=(24, 96),
        supports_static=True,
        typical_training_time_minutes=15.0,
        memory_footprint_mb=1000.0
    ),
    # 複雑なモデル
    "TFT": ModelCharacteristics(
        name="TFT",
        complexity=ModelComplexity.COMPLEX,
        recommended_input_size_range=(24, 168),
        supports_static=True,
        typical_training_time_minutes=30.0,
        memory_footprint_mb=2000.0
    ),
    "Transformer": ModelCharacteristics(
        name="Transformer",
        complexity=ModelComplexity.COMPLEX,
        recommended_input_size_range=(24, 96),
        typical_training_time_minutes=25.0,
        memory_footprint_mb=1500.0
    ),
    "PatchTST": ModelCharacteristics(
        name="PatchTST",
        complexity=ModelComplexity.COMPLEX,
        recommended_input_size_range=(96, 512),
        typical_training_time_minutes=20.0,
        memory_footprint_mb=1200.0
    ),
}


@dataclass
class OptimizationConfig:
    """最適化設定"""
    # バックエンド設定
    backend: str = "optuna"
    num_samples: Optional[int] = None  # Noneの場合は自動推奨
    cpus: int = 4
    gpus: int = 0
    
    # 検証設定
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # MLflow設定
    use_mlflow: bool = True
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    
    # 最適化設定
    use_pruning: bool = True
    random_seed: Optional[int] = 42
    time_budget_hours: Optional[float] = None
    
    # ロギング
    verbose: bool = True
    callbacks: Optional[List[Callable]] = None
    
    # 検証オプション
    strict_validation: bool = False
    auto_correct: bool = True


class AutoModelFactory:
    """
    自動モデルファクトリー
    NeuralForecastモデルの自動ハイパーパラメータ最適化を管理
    """
    
    def __init__(
        self,
        model_name: str,
        h: int,
        optimization_config: Optional[OptimizationConfig] = None
    ):
        """
        Args:
            model_name: モデル名（例: "NHITS", "TFT"）
            h: 予測ホライゾン
            optimization_config: 最適化設定
        """
        self.model_name = model_name
        self.h = h
        self.config = optimization_config or OptimizationConfig()
        
        # モデル特性の取得
        if model_name not in MODEL_CATALOG:
            logger.warning(
                f"Model '{model_name}' not in catalog. Using default characteristics."
            )
            self.model_char = ModelCharacteristics(
                name=model_name,
                complexity=ModelComplexity.MODERATE
            )
        else:
            self.model_char = MODEL_CATALOG[model_name]
        
        logger.info(f"Initialized AutoModelFactory for {model_name}")
        logger.info(f"Model characteristics: {self.model_char}")
        
        # コンポーネント初期化
        self.config_validator = ConfigValidator(
            strict_mode=self.config.strict_validation
        )
        self.data_validator = DataValidator()
        self.algorithm_selector = SearchAlgorithmSelector(
            backend=self.config.backend
        )
        
        # MLflow設定
        if self.config.use_mlflow:
            self._setup_mlflow()
        
        # 最適化履歴
        self.optimization_history = []
    
    def _setup_mlflow(self):
        """MLflowのセットアップ"""
        try:
            import mlflow
            
            if self.config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            
            if self.config.mlflow_experiment_name:
                mlflow.set_experiment(self.config.mlflow_experiment_name)
            else:
                experiment_name = f"neuralforecast_{self.model_name}_{datetime.now().strftime('%Y%m%d')}"
                mlflow.set_experiment(experiment_name)
            
            logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            logger.info(f"MLflow experiment: {mlflow.get_experiment(mlflow.active_experiment_id()).name}")
        
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            self.config.use_mlflow = False
    
    def create_auto_model(
        self,
        dataset: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        loss: Optional[Any] = None,
        valid_loss: Optional[Any] = None,
        **kwargs
    ):
        """
        自動最適化モデルを作成
        
        Args:
            dataset: 学習データ
            config: ハイパーパラメータ探索空間（Noneの場合はデフォルト）
            loss: 損失関数
            valid_loss: 検証損失関数
            **kwargs: その他のモデルパラメータ
            
        Returns:
            NeuralForecastのAutoモデル: 最適化済みモデル
        """
        start_time = time.time()
        
        # ステップ1: 検証
        logger.info("=" * 60)
        logger.info("STEP 1: Validation")
        logger.info("=" * 60)
        
        validation_results = self._validate_configuration(
            dataset=dataset,
            config=config or {}
        )
        
        if not all(result.is_valid for result in validation_results.values()):
            if self.config.strict_validation:
                raise ValueError("Validation failed. See logs for details.")
            else:
                logger.warning("Some validations failed but continuing...")
        
        # ステップ2: データセット分析
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Dataset Analysis")
        logger.info("=" * 60)
        
        dataset_info = self._analyze_dataset(dataset)
        
        # ステップ3: 探索戦略の選択
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Search Strategy Selection")
        logger.info("=" * 60)
        
        search_strategy = self._select_search_strategy(
            dataset_info=dataset_info,
            config=config or {}
        )
        
        # ステップ4: ハイパーパラメータ設定の準備
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Hyperparameter Configuration")
        logger.info("=" * 60)
        
        final_config = self._prepare_config(config, dataset_info)
        
        # ステップ5: モデルの作成と最適化
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Model Creation and Optimization")
        logger.info("=" * 60)
        
        auto_model = self._create_and_optimize(
            dataset=dataset,
            config=final_config,
            search_strategy=search_strategy,
            loss=loss,
            valid_loss=valid_loss,
            **kwargs
        )
        
        # 最適化履歴の記録
        elapsed_time = time.time() - start_time
        self._record_optimization(
            elapsed_time=elapsed_time,
            dataset_info=dataset_info,
            search_strategy=search_strategy,
            final_config=final_config
        )
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✓ Optimization completed in {elapsed_time:.2f} seconds")
        logger.info("=" * 60)
        
        return auto_model
    
    def _validate_configuration(
        self,
        dataset: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """設定の検証"""
        validation_results = validate_all(
            backend=self.config.backend,
            config=config,
            num_samples=self.config.num_samples or 50,
            cpus=self.config.cpus,
            gpus=self.config.gpus,
            model_class_name=self.model_name,
            dataset=dataset,
            h=self.h,
            mlflow_tracking_uri=self.config.mlflow_tracking_uri,
            mlflow_experiment_name=self.config.mlflow_experiment_name,
            strict_mode=self.config.strict_validation
        )
        
        if self.config.verbose:
            print_validation_results(validation_results)
        
        return validation_results
    
    def _analyze_dataset(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """データセットの分析"""
        n_series = dataset['unique_id'].nunique()
        n_observations = len(dataset)
        avg_length = n_observations / n_series
        
        # データセットサイズの判定
        if n_observations < 1000:
            dataset_size = DatasetSize.SMALL
        elif n_observations < 100000:
            dataset_size = DatasetSize.MEDIUM
        else:
            dataset_size = DatasetSize.LARGE
        
        # 特徴量の確認
        has_exog = any(col not in ['unique_id', 'ds', 'y'] for col in dataset.columns)
        
        info = {
            'n_series': n_series,
            'n_observations': n_observations,
            'avg_length': avg_length,
            'dataset_size': dataset_size,
            'has_exog': has_exog,
            'columns': list(dataset.columns)
        }
        
        logger.info(f"Dataset info:")
        logger.info(f"  - Series: {n_series}")
        logger.info(f"  - Observations: {n_observations}")
        logger.info(f"  - Avg length per series: {avg_length:.1f}")
        logger.info(f"  - Size category: {dataset_size.value}")
        logger.info(f"  - Has exogenous variables: {has_exog}")
        
        return info
    
    def _select_search_strategy(
        self,
        dataset_info: Dict[str, Any],
        config: Dict[str, Any]
    ):
        """探索戦略の選択"""
        # num_samplesの決定
        if self.config.num_samples is None:
            recommended, explanation = recommend_num_samples(
                model_complexity=self.model_char.complexity,
                dataset_size=dataset_info['dataset_size'],
                search_complexity=SearchComplexity.MEDIUM,  # デフォルト
                time_budget_hours=self.config.time_budget_hours
            )
            self.config.num_samples = recommended
            logger.info(f"Auto-selected num_samples: {explanation}")
        
        # 探索アルゴリズムの選択
        strategy = self.algorithm_selector.select_algorithm(
            model_complexity=self.model_char.complexity,
            dataset_size=dataset_info['dataset_size'],
            num_samples=self.config.num_samples,
            config=config,
            use_pruning=self.config.use_pruning,
            random_seed=self.config.random_seed
        )
        
        logger.info(f"Selected search strategy: {strategy}")
        
        return strategy
    
    def _prepare_config(
        self,
        config: Optional[Dict[str, Any]],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """設定の準備"""
        if config is not None:
            return config
        
        # デフォルト設定の生成
        logger.info("Using default configuration for hyperparameter search")
        
        # NeuralForecastのインポート
        from ray import tune
        
        # 基本的なハイパーパラメータ空間
        default_config = {
            'max_steps': tune.choice([500, 1000, 2000, 3000]),
            'learning_rate': tune.loguniform(1e-4, 1e-2),
            'batch_size': tune.choice([32, 64, 128, 256]),
            'val_check_steps': tune.choice([50, 100, 200]),
            'early_stop_patience_steps': tune.choice([2, 3, 5]),
        }
        
        # モデル固有のパラメータ追加
        if self.model_name in ["NHITS", "NBEATS"]:
            default_config.update({
                'n_blocks': tune.choice([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
                'mlp_units': tune.choice([[[512, 512]], [[256, 256]], [[128, 128]]]),
            })
        elif self.model_name == "TFT":
            default_config.update({
                'hidden_size': tune.choice([64, 128, 256]),
                'dropout': tune.uniform(0.1, 0.3),
                'num_attention_heads': tune.choice([2, 4, 8]),
            })
        elif self.model_name == "Transformer":
            default_config.update({
                'n_head': tune.choice([2, 4, 8]),
                'hidden_size': tune.choice([64, 128, 256]),
                'dropout': tune.uniform(0.1, 0.3),
            })
        
        # input_sizeの推奨
        min_input, max_input = self.model_char.recommended_input_size_range
        avg_length = dataset_info['avg_length']
        
        if avg_length < min_input:
            logger.warning(
                f"Average series length ({avg_length:.1f}) is less than "
                f"recommended minimum input_size ({min_input})"
            )
            input_size_options = [int(avg_length * 0.5), int(avg_length * 0.7)]
        else:
            input_size_options = [
                size for size in [min_input, (min_input + max_input) // 2, max_input]
                if size < avg_length
            ]
        
        if input_size_options:
            default_config['input_size'] = tune.choice(input_size_options)
        
        logger.info(f"Generated default config with {len(default_config)} parameters")
        
        return default_config
    
    def _create_and_optimize(
        self,
        dataset: pd.DataFrame,
        config: Dict[str, Any],
        search_strategy: Any,
        loss: Optional[Any],
        valid_loss: Optional[Any],
        **kwargs
    ):
        """モデルの作成と最適化"""
        # NeuralForecastのAutoモデルをインポート
        # 動的インポート（モデル名に基づく）
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.auto import (
                AutoNHITS, AutoNBEATS, AutoTFT, 
                AutoMLP, AutoDLinear, AutoTSMixer,
                AutoPatchTST, AutoVanillaTransformer
            )
            
            auto_model_map = {
                'NHITS': AutoNHITS,
                'NBEATS': AutoNBEATS,
                'TFT': AutoTFT,
                'MLP': AutoMLP,
                'DLinear': AutoDLinear,
                'TSMixer': AutoTSMixer,
                'PatchTST': AutoPatchTST,
                'Transformer': AutoVanillaTransformer,
            }
            
            AutoModelClass = auto_model_map.get(self.model_name)
            if AutoModelClass is None:
                raise ValueError(f"Auto model not found for {self.model_name}")
            
        except ImportError as e:
            logger.error(f"Failed to import NeuralForecast Auto models: {e}")
            raise
        
        # Optunaバックエンドの場合
        if self.config.backend == "optuna":
            sampler = self.algorithm_selector.get_optuna_sampler(search_strategy)
            pruner = self.algorithm_selector.get_optuna_pruner(search_strategy)
            
            # search_algパラメータを構築
            import optuna
            
            # Studyの作成
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner
            )
            
            search_alg = study
        else:
            # Rayバックエンド
            search_alg = search_strategy
        
        # n_seriesの計算
        n_series = dataset['unique_id'].nunique()
        
        # MLflow実験の開始
        if self.config.use_mlflow:
            import mlflow
            with mlflow.start_run(run_name=f"{self.model_name}_auto_optimization"):
                # パラメータのログ
                mlflow.log_params({
                    'model_name': self.model_name,
                    'h': self.h,
                    'backend': self.config.backend,
                    'num_samples': self.config.num_samples,
                    'n_series': n_series
                })
                
                # Autoモデルの作成
                auto_model = AutoModelClass(
                    h=self.h,
                    n_series=n_series,
                    config=config,
                    loss=loss,
                    valid_loss=valid_loss,
                    num_samples=self.config.num_samples,
                    cpus=self.config.cpus,
                    gpus=self.config.gpus,
                    backend=self.config.backend,
                    search_alg=search_alg,
                    verbose=self.config.verbose,
                    callbacks=self.config.callbacks,
                    **kwargs
                )
                
                # 学習
                logger.info(f"Starting optimization with {self.config.num_samples} trials...")
                auto_model.fit(dataset=dataset)
                
                # 最良の設定をログ
                if hasattr(auto_model, 'results_'):
                    mlflow.log_metrics({
                        'best_loss': float(auto_model.results_.get_best_result().metrics.get('loss', float('inf')))
                    })
        else:
            # MLflowなしでの実行
            auto_model = AutoModelClass(
                h=self.h,
                n_series=n_series,
                config=config,
                loss=loss,
                valid_loss=valid_loss,
                num_samples=self.config.num_samples,
                cpus=self.config.cpus,
                gpus=self.config.gpus,
                backend=self.config.backend,
                search_alg=search_alg,
                verbose=self.config.verbose,
                callbacks=self.config.callbacks,
                **kwargs
            )
            
            logger.info(f"Starting optimization with {self.config.num_samples} trials...")
            auto_model.fit(dataset=dataset)
        
        return auto_model
    
    def _record_optimization(
        self,
        elapsed_time: float,
        dataset_info: Dict[str, Any],
        search_strategy: Any,
        final_config: Dict[str, Any]
    ):
        """最適化履歴の記録"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'elapsed_time_seconds': elapsed_time,
            'dataset_info': dataset_info,
            'search_strategy': str(search_strategy),
            'num_samples': self.config.num_samples,
            'backend': self.config.backend
        }
        
        self.optimization_history.append(record)
        
        if self.config.verbose:
            logger.info("Optimization summary:")
            logger.info(json.dumps(record, indent=2, default=str))
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """最適化履歴のサマリーを取得"""
        if not self.optimization_history:
            logger.warning("No optimization history available")
            return pd.DataFrame()
        
        return pd.DataFrame(self.optimization_history)


# 便利関数
def create_auto_model(
    model_name: str,
    h: int,
    dataset: pd.DataFrame,
    backend: str = "optuna",
    num_samples: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    cpus: int = 4,
    gpus: int = 0,
    use_mlflow: bool = True,
    mlflow_experiment_name: Optional[str] = None,
    verbose: bool = True,
    **kwargs
):
    """
    自動最適化モデルを作成する便利関数
    
    Args:
        model_name: モデル名
        h: 予測ホライゾン
        dataset: 学習データ
        backend: 最適化バックエンド
        num_samples: 試行回数
        config: ハイパーパラメータ探索空間
        cpus: CPU数
        gpus: GPU数
        use_mlflow: MLflowを使用するか
        mlflow_experiment_name: MLflow実験名
        verbose: 詳細出力
        **kwargs: その他のパラメータ
        
    Returns:
        最適化済みAutoモデル
    """
    opt_config = OptimizationConfig(
        backend=backend,
        num_samples=num_samples,
        cpus=cpus,
        gpus=gpus,
        use_mlflow=use_mlflow,
        mlflow_experiment_name=mlflow_experiment_name,
        verbose=verbose
    )
    
    factory = AutoModelFactory(
        model_name=model_name,
        h=h,
        optimization_config=opt_config
    )
    
    return factory.create_auto_model(
        dataset=dataset,
        config=config,
        **kwargs
    )
