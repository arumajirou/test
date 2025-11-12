# src/search_algorithm_selector.py
"""
探索アルゴリズム選択モジュール
モデル特性、データセット特性、計算リソースに基づいて
最適なハイパーパラメータ探索アルゴリズムを選択
"""
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SearchComplexity(Enum):
    """探索空間の複雑度"""
    LOW = "low"          # < 5パラメータ、小さな範囲
    MEDIUM = "medium"    # 5-15パラメータ、中程度の範囲
    HIGH = "high"        # > 15パラメータ、大きな範囲


class ModelComplexity(Enum):
    """モデルの複雑度"""
    SIMPLE = "simple"        # MLP, Linear等
    MODERATE = "moderate"    # NHITS, DLinear等
    COMPLEX = "complex"      # TFT, Transformer等


class DatasetSize(Enum):
    """データセットサイズ"""
    SMALL = "small"      # < 1000行
    MEDIUM = "medium"    # 1000-100000行
    LARGE = "large"      # > 100000行


@dataclass
class SearchStrategy:
    """探索戦略の定義"""
    algorithm_name: str
    sampler_kwargs: Dict[str, Any]
    pruner_name: Optional[str] = None
    pruner_kwargs: Optional[Dict[str, Any]] = None
    description: str = ""
    
    def __repr__(self):
        return (
            f"SearchStrategy(algorithm={self.algorithm_name}, "
            f"pruner={self.pruner_name})"
        )


class SearchAlgorithmSelector:
    """探索アルゴリズム選択クラス"""
    
    def __init__(self, backend: str = "optuna"):
        """
        Args:
            backend: 使用するバックエンド ('ray' or 'optuna')
        """
        self.backend = backend.lower()
        if self.backend not in ["ray", "optuna"]:
            raise ValueError(f"Invalid backend: {backend}")
    
    def select_algorithm(
        self,
        model_complexity: Union[ModelComplexity, str],
        dataset_size: Union[DatasetSize, str, int],
        num_samples: int,
        config: Dict[str, Any],
        use_pruning: bool = True,
        random_seed: Optional[int] = None
    ) -> Union[SearchStrategy, Any]:
        """
        最適な探索アルゴリズムを選択
        
        Args:
            model_complexity: モデルの複雑度
            dataset_size: データセットサイズ
            num_samples: 試行回数
            config: ハイパーパラメータ設定
            use_pruning: プルーニングを使用するか
            random_seed: 乱数シード
            
        Returns:
            SearchStrategy or Ray search_alg: 選択されたアルゴリズム
        """
        # Enum変換
        if isinstance(model_complexity, str):
            model_complexity = ModelComplexity(model_complexity.lower())
        
        if isinstance(dataset_size, int):
            if dataset_size < 1000:
                dataset_size = DatasetSize.SMALL
            elif dataset_size < 100000:
                dataset_size = DatasetSize.MEDIUM
            else:
                dataset_size = DatasetSize.LARGE
        elif isinstance(dataset_size, str):
            dataset_size = DatasetSize(dataset_size.lower())
        
        # 探索空間の複雑度を推定
        search_complexity = self._estimate_search_complexity(config)
        
        logger.info(
            f"Selecting algorithm: model={model_complexity.value}, "
            f"data={dataset_size.value}, search={search_complexity.value}, "
            f"samples={num_samples}"
        )
        
        if self.backend == "optuna":
            return self._select_optuna_algorithm(
                model_complexity=model_complexity,
                dataset_size=dataset_size,
                search_complexity=search_complexity,
                num_samples=num_samples,
                use_pruning=use_pruning,
                random_seed=random_seed
            )
        else:  # ray
            return self._select_ray_algorithm(
                model_complexity=model_complexity,
                dataset_size=dataset_size,
                search_complexity=search_complexity,
                num_samples=num_samples,
                random_seed=random_seed
            )
    
    def _estimate_search_complexity(
        self,
        config: Dict[str, Any]
    ) -> SearchComplexity:
        """
        探索空間の複雑度を推定
        
        Args:
            config: ハイパーパラメータ設定
            
        Returns:
            SearchComplexity: 推定された複雑度
        """
        # パラメータ数をカウント
        param_count = len(config)
        
        # 範囲の広さを推定（Rayの場合）
        try:
            from ray import tune
            continuous_params = 0
            categorical_params = 0
            
            for value in config.values():
                if hasattr(value, 'sampler'):
                    if isinstance(value, tune.search.sample.Categorical):
                        categorical_params += 1
                    else:
                        continuous_params += 1
            
            # カテゴリカル変数が多い場合は複雑度が上がる
            if categorical_params > 5:
                return SearchComplexity.HIGH
        except ImportError:
            pass
        
        # パラメータ数に基づく判定
        if param_count < 5:
            return SearchComplexity.LOW
        elif param_count < 15:
            return SearchComplexity.MEDIUM
        else:
            return SearchComplexity.HIGH
    
    def _select_optuna_algorithm(
        self,
        model_complexity: ModelComplexity,
        dataset_size: DatasetSize,
        search_complexity: SearchComplexity,
        num_samples: int,
        use_pruning: bool,
        random_seed: Optional[int]
    ) -> SearchStrategy:
        """
        Optuna用のアルゴリズムを選択
        
        Returns:
            SearchStrategy: 選択された戦略
        """
        # デフォルト設定
        sampler_kwargs = {"seed": random_seed} if random_seed else {}
        pruner_name = None
        pruner_kwargs = None
        
        # 試行回数が少ない場合
        if num_samples < 10:
            algorithm = "RandomSampler"
            description = "Few trials: using random sampling"
        
        # 探索空間が小さい場合
        elif search_complexity == SearchComplexity.LOW and num_samples < 50:
            algorithm = "GridSampler"
            description = "Small search space: using grid search"
        
        # 大規模データセット + 複雑なモデル
        elif (dataset_size == DatasetSize.LARGE and 
              model_complexity == ModelComplexity.COMPLEX):
            algorithm = "TPESampler"
            sampler_kwargs.update({
                "n_startup_trials": min(20, num_samples // 4),
                "n_ei_candidates": 24,
                "multivariate": True
            })
            description = "Large dataset + complex model: TPE with multivariate"
            
            if use_pruning:
                pruner_name = "HyperbandPruner"
                pruner_kwargs = {
                    "min_resource": 5,
                    "max_resource": 100,
                    "reduction_factor": 3
                }
        
        # 中規模データセット
        elif dataset_size == DatasetSize.MEDIUM:
            if search_complexity == SearchComplexity.HIGH:
                # 高次元探索空間：CMA-ES
                algorithm = "CmaEsSampler"
                sampler_kwargs.update({
                    "n_startup_trials": min(10, num_samples // 5)
                })
                description = "Medium data + high complexity: CMA-ES"
            else:
                # 標準的なTPE
                algorithm = "TPESampler"
                sampler_kwargs.update({
                    "n_startup_trials": min(10, num_samples // 5)
                })
                description = "Medium dataset: standard TPE"
            
            if use_pruning and model_complexity == ModelComplexity.COMPLEX:
                pruner_name = "MedianPruner"
                pruner_kwargs = {
                    "n_startup_trials": 5,
                    "n_warmup_steps": 10
                }
        
        # 小規模データセット
        else:  # DatasetSize.SMALL
            if num_samples > 50:
                # 多数の試行：TPE
                algorithm = "TPESampler"
                sampler_kwargs.update({
                    "n_startup_trials": 10
                })
                description = "Small dataset, many trials: TPE"
            else:
                # 少数の試行：ランダムサンプリング
                algorithm = "RandomSampler"
                description = "Small dataset, few trials: random sampling"
            
            # 小規模データではプルーニングは慎重に
            if use_pruning and model_complexity == ModelComplexity.COMPLEX:
                pruner_name = "MedianPruner"
                pruner_kwargs = {
                    "n_startup_trials": 10,
                    "n_warmup_steps": 20,
                    "interval_steps": 5
                }
        
        logger.info(f"Selected Optuna strategy: {description}")
        
        return SearchStrategy(
            algorithm_name=algorithm,
            sampler_kwargs=sampler_kwargs,
            pruner_name=pruner_name,
            pruner_kwargs=pruner_kwargs,
            description=description
        )
    
    def _select_ray_algorithm(
        self,
        model_complexity: ModelComplexity,
        dataset_size: DatasetSize,
        search_complexity: SearchComplexity,
        num_samples: int,
        random_seed: Optional[int]
    ) -> Any:
        """
        Ray Tune用のアルゴリズムを選択
        
        Returns:
            Ray search_alg: 選択されたアルゴリズム
        """
        from ray.tune.search import BasicVariantGenerator
        
        try:
            # 試行回数が少ない場合：基本的なランダム探索
            if num_samples < 10:
                logger.info("Few trials: using BasicVariantGenerator")
                return BasicVariantGenerator(random_state=random_seed)
            
            # 探索空間が小さい場合：グリッドサーチ
            if search_complexity == SearchComplexity.LOW and num_samples < 50:
                logger.info("Small search space: using BasicVariantGenerator (grid-like)")
                return BasicVariantGenerator(random_state=random_seed)
            
            # 大規模データセット + 複雑なモデル：OptunaSearch with TPE
            if (dataset_size == DatasetSize.LARGE and 
                model_complexity == ModelComplexity.COMPLEX):
                try:
                    from ray.tune.search.optuna import OptunaSearch
                    import optuna
                    
                    sampler = optuna.samplers.TPESampler(
                        seed=random_seed,
                        n_startup_trials=min(20, num_samples // 4),
                        n_ei_candidates=24,
                        multivariate=True
                    )
                    
                    logger.info("Large dataset + complex model: OptunaSearch with TPE")
                    return OptunaSearch(sampler=sampler)
                except ImportError:
                    logger.warning("OptunaSearch not available, falling back to BasicVariantGenerator")
                    return BasicVariantGenerator(random_state=random_seed)
            
            # 中規模データセット：BayesOptSearch
            if dataset_size == DatasetSize.MEDIUM:
                try:
                    from ray.tune.search.bayesopt import BayesOptSearch
                    
                    logger.info("Medium dataset: using BayesOptSearch")
                    return BayesOptSearch(
                        random_state=random_seed,
                        random_search_steps=min(5, num_samples // 10)
                    )
                except ImportError:
                    logger.warning("BayesOptSearch not available, falling back")
                    # OptunaSearchへフォールバック
                    try:
                        from ray.tune.search.optuna import OptunaSearch
                        logger.info("Using OptunaSearch as fallback")
                        return OptunaSearch()
                    except ImportError:
                        return BasicVariantGenerator(random_state=random_seed)
            
            # デフォルト：HyperOptSearch
            try:
                from ray.tune.search.hyperopt import HyperOptSearch
                
                logger.info("Default: using HyperOptSearch")
                return HyperOptSearch(
                    random_state_seed=random_seed,
                    n_initial_points=min(10, num_samples // 5)
                )
            except ImportError:
                logger.warning("HyperOptSearch not available, using BasicVariantGenerator")
                return BasicVariantGenerator(random_state=random_seed)
        
        except Exception as e:
            logger.error(f"Error selecting Ray algorithm: {e}")
            logger.info("Falling back to BasicVariantGenerator")
            return BasicVariantGenerator(random_state=random_seed)
    
    def get_optuna_sampler(
        self,
        strategy: SearchStrategy
    ):
        """
        Optuna samplerオブジェクトを取得
        
        Args:
            strategy: SearchStrategy
            
        Returns:
            optuna.samplers.BaseSampler: サンプラーオブジェクト
        """
        if self.backend != "optuna":
            raise ValueError("This method is only for Optuna backend")
        
        import optuna
        
        sampler_map = {
            "TPESampler": optuna.samplers.TPESampler,
            "CmaEsSampler": optuna.samplers.CmaEsSampler,
            "RandomSampler": optuna.samplers.RandomSampler,
            "GridSampler": optuna.samplers.GridSampler,
            "BruteForceSampler": optuna.samplers.BruteForceSampler
        }
        
        sampler_class = sampler_map.get(strategy.algorithm_name)
        if sampler_class is None:
            logger.warning(
                f"Unknown sampler: {strategy.algorithm_name}, using TPESampler"
            )
            sampler_class = optuna.samplers.TPESampler
        
        try:
            return sampler_class(**strategy.sampler_kwargs)
        except Exception as e:
            logger.error(f"Error creating sampler: {e}")
            logger.info("Falling back to TPESampler with default config")
            return optuna.samplers.TPESampler()
    
    def get_optuna_pruner(
        self,
        strategy: SearchStrategy
    ):
        """
        Optuna prunerオブジェクトを取得
        
        Args:
            strategy: SearchStrategy
            
        Returns:
            optuna.pruners.BasePruner or None: プルーナーオブジェクト
        """
        if self.backend != "optuna":
            raise ValueError("This method is only for Optuna backend")
        
        if strategy.pruner_name is None:
            return None
        
        import optuna
        
        pruner_map = {
            "MedianPruner": optuna.pruners.MedianPruner,
            "HyperbandPruner": optuna.pruners.HyperbandPruner,
            "SuccessiveHalvingPruner": optuna.pruners.SuccessiveHalvingPruner,
            "PercentilePruner": optuna.pruners.PercentilePruner
        }
        
        pruner_class = pruner_map.get(strategy.pruner_name)
        if pruner_class is None:
            logger.warning(
                f"Unknown pruner: {strategy.pruner_name}, using MedianPruner"
            )
            pruner_class = optuna.pruners.MedianPruner
        
        try:
            kwargs = strategy.pruner_kwargs or {}
            return pruner_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating pruner: {e}")
            logger.info("Falling back to MedianPruner with default config")
            return optuna.pruners.MedianPruner()


def recommend_num_samples(
    model_complexity: Union[ModelComplexity, str],
    dataset_size: Union[DatasetSize, str, int],
    search_complexity: Union[SearchComplexity, str],
    time_budget_hours: Optional[float] = None
) -> Tuple[int, str]:
    """
    推奨される試行回数を計算
    
    Args:
        model_complexity: モデルの複雑度
        dataset_size: データセットサイズ
        search_complexity: 探索空間の複雑度
        time_budget_hours: 時間予算（時間）
        
    Returns:
        Tuple[int, str]: (推奨試行回数, 説明)
    """
    # Enum変換
    if isinstance(model_complexity, str):
        model_complexity = ModelComplexity(model_complexity.lower())
    if isinstance(dataset_size, int):
        if dataset_size < 1000:
            dataset_size = DatasetSize.SMALL
        elif dataset_size < 100000:
            dataset_size = DatasetSize.MEDIUM
        else:
            dataset_size = DatasetSize.LARGE
    elif isinstance(dataset_size, str):
        dataset_size = DatasetSize(dataset_size.lower())
    if isinstance(search_complexity, str):
        search_complexity = SearchComplexity(search_complexity.lower())
    
    # 基本的な推奨値
    base_samples = {
        (SearchComplexity.LOW, ModelComplexity.SIMPLE): 10,
        (SearchComplexity.LOW, ModelComplexity.MODERATE): 15,
        (SearchComplexity.LOW, ModelComplexity.COMPLEX): 20,
        (SearchComplexity.MEDIUM, ModelComplexity.SIMPLE): 20,
        (SearchComplexity.MEDIUM, ModelComplexity.MODERATE): 30,
        (SearchComplexity.MEDIUM, ModelComplexity.COMPLEX): 50,
        (SearchComplexity.HIGH, ModelComplexity.SIMPLE): 50,
        (SearchComplexity.HIGH, ModelComplexity.MODERATE): 100,
        (SearchComplexity.HIGH, ModelComplexity.COMPLEX): 150,
    }
    
    recommended = base_samples.get(
        (search_complexity, model_complexity),
        50  # デフォルト
    )
    
    # データセットサイズによる調整
    if dataset_size == DatasetSize.SMALL:
        recommended = int(recommended * 0.7)
    elif dataset_size == DatasetSize.LARGE:
        recommended = int(recommended * 1.3)
    
    # 時間予算による調整
    if time_budget_hours:
        # 1試行あたりの平均時間を推定（分）
        avg_trial_minutes = {
            (DatasetSize.SMALL, ModelComplexity.SIMPLE): 2,
            (DatasetSize.SMALL, ModelComplexity.MODERATE): 5,
            (DatasetSize.SMALL, ModelComplexity.COMPLEX): 10,
            (DatasetSize.MEDIUM, ModelComplexity.SIMPLE): 5,
            (DatasetSize.MEDIUM, ModelComplexity.MODERATE): 15,
            (DatasetSize.MEDIUM, ModelComplexity.COMPLEX): 30,
            (DatasetSize.LARGE, ModelComplexity.SIMPLE): 15,
            (DatasetSize.LARGE, ModelComplexity.MODERATE): 45,
            (DatasetSize.LARGE, ModelComplexity.COMPLEX): 90,
        }.get((dataset_size, model_complexity), 15)
        
        budget_based_samples = int(
            (time_budget_hours * 60) / avg_trial_minutes
        )
        
        if budget_based_samples < recommended:
            explanation = (
                f"Time budget limits to {budget_based_samples} trials "
                f"(originally recommended {recommended})"
            )
            recommended = budget_based_samples
        else:
            explanation = (
                f"Recommended {recommended} trials "
                f"(time budget allows up to {budget_based_samples})"
            )
    else:
        explanation = (
            f"Recommended {recommended} trials based on "
            f"model complexity ({model_complexity.value}), "
            f"dataset size ({dataset_size.value}), and "
            f"search complexity ({search_complexity.value})"
        )
    
    return recommended, explanation


# エクスポート用の便利関数
def create_selector(backend: str = "optuna") -> SearchAlgorithmSelector:
    """
    SearchAlgorithmSelectorのファクトリー関数
    
    Args:
        backend: バックエンド ('ray' or 'optuna')
        
    Returns:
        SearchAlgorithmSelector: セレクターインスタンス
    """
    return SearchAlgorithmSelector(backend=backend)
