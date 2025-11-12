"""
NeuralForecast AutoModels の特性定義

このモジュールは、全AutoModelsの探索空間特性、パラメータ設定、
最適化アルゴリズムの推奨設定を一元管理します。

互換表（アップロードファイル）から抽出した情報を基に、
各モデルに最適なsearch_alg、pruner、schedulerを決定する
ための判断材料を提供します。

Example:
    >>> from model_characteristics import MODEL_CHARACTERISTICS
    >>> char = MODEL_CHARACTERISTICS["AutoTFT"]
    >>> print(char.dropout_key)  # "dropout"
    >>> print(char.default_scaler)  # "identity"
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class OptimizationType(str, Enum):
    """最適化空間のタイプ"""
    CONTINUOUS = "continuous"          # 連続空間のみ
    DISCRETE = "discrete"              # 離散空間のみ
    MIXED = "mixed"                    # 連続+離散の混合
    MULTI_OBJECTIVE = "multi_objective"  # 多目的最適化


class ExogenousSupport(str, Enum):
    """外生変数のサポート種別"""
    NONE = "none"           # 外生変数非対応
    FUTR_ONLY = "futr_only"  # 未来外生のみ
    HIST_ONLY = "hist_only"  # 履歴外生のみ
    STAT_ONLY = "stat_only"  # 静的外生のみ
    FULL = "full"           # 全種類対応


@dataclass
class ModelCharacteristics:
    """
    AutoModelの特性を表現するデータクラス
    
    このクラスは、SearchAlgorithmSelectorが最適な
    検索アルゴリズムを選択するための判断材料を提供します。
    
    Attributes:
        model_name: モデル名（例: "AutoTFT"）
        base_model_name: ベースモデル名（例: "TFT"）
        has_continuous_params: 連続パラメータの有無
        has_discrete_params: 離散パラメータの有無
        has_categorical_params: カテゴリカルパラメータの有無
        supports_multi_objective: 多目的最適化対応
        typical_search_space_size: 探索空間のおおよそのサイズ
        optimization_type: 最適化タイプ（自動推論）
        dropout_supported: ドロップアウト対応
        dropout_key: ドロップアウトパラメータのキー名
        dropout_default: ドロップアウトのデフォルト値
        default_scaler: デフォルトスケーラー種別
        scaler_choices: 選択可能なスケーラー
        exogenous_support: 外生変数サポート種別
        early_stop_default: 早期停止のデフォルト設定
        backend_choices: 選択可能なbackend
        pruner_default_enabled_optuna: Optunaでprunerがデフォルト有効か
        notes: 追加メモ
    """
    model_name: str
    base_model_name: str
    has_continuous_params: bool = True
    has_discrete_params: bool = True
    has_categorical_params: bool = True
    supports_multi_objective: bool = False
    typical_search_space_size: int = 10000
    
    # ドロップアウト
    dropout_supported: bool = False
    dropout_key: Optional[str] = None
    dropout_default: Optional[float] = None
    
    # スケーラー
    default_scaler: str = "identity"
    scaler_choices: List[str] = field(default_factory=lambda: ["identity", "standard", "robust", "minmax"])
    
    # 外生変数
    exogenous_support: ExogenousSupport = ExogenousSupport.FULL
    
    # 早期停止
    early_stop_default: int = -1  # -1は無効
    
    # Backend/Pruner
    backend_choices: List[str] = field(default_factory=lambda: ["ray", "optuna"])
    pruner_default_enabled_optuna: bool = True
    
    # その他
    notes: str = ""
    
    @property
    def optimization_type(self) -> OptimizationType:
        """最適化タイプを自動推論"""
        if self.supports_multi_objective:
            return OptimizationType.MULTI_OBJECTIVE
        elif self.has_continuous_params and not self.has_discrete_params:
            return OptimizationType.CONTINUOUS
        elif not self.has_continuous_params and self.has_discrete_params:
            return OptimizationType.DISCRETE
        else:
            return OptimizationType.MIXED


# =========================================================================
# 全AutoModelsの特性定義（互換表から抽出）
# =========================================================================

MODEL_CHARACTERISTICS: Dict[str, ModelCharacteristics] = {
    
    # --- Transformer系 ---
    "AutoAutoformer": ModelCharacteristics(
        model_name="AutoAutoformer",
        base_model_name="Autoformer",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=15000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.05,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Decomposition-based transformer, good with trend/seasonality"
    ),
    
    "AutoTFT": ModelCharacteristics(
        model_name="AutoTFT",
        base_model_name="TFT",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=20000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.1,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Temporal Fusion Transformer, excellent with exogenous variables"
    ),
    
    "AutoPatchTST": ModelCharacteristics(
        model_name="AutoPatchTST",
        base_model_name="PatchTST",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=12000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.2,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.NONE,
        notes="Patch-based transformer, efficient for long sequences"
    ),
    
    "AutoVanillaTransformer": ModelCharacteristics(
        model_name="AutoVanillaTransformer",
        base_model_name="VanillaTransformer",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=10000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.1,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL
    ),
    
    "AutoInformer": ModelCharacteristics(
        model_name="AutoInformer",
        base_model_name="Informer",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=15000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.05,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="ProbSparse attention for long-range dependencies"
    ),
    
    "AutoFEDformer": ModelCharacteristics(
        model_name="AutoFEDformer",
        base_model_name="FEDformer",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=15000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.05,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Frequency Enhanced Decomposition Transformer"
    ),
    
    "AutoiTransformer": ModelCharacteristics(
        model_name="AutoiTransformer",
        base_model_name="iTransformer",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=12000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.1,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Inverted transformer architecture"
    ),
    
    # --- RNN系 ---
    "AutoRNN": ModelCharacteristics(
        model_name="AutoRNN",
        base_model_name="RNN",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=5000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.0,
        default_scaler="standard",
        exogenous_support=ExogenousSupport.FULL
    ),
    
    "AutoLSTM": ModelCharacteristics(
        model_name="AutoLSTM",
        base_model_name="LSTM",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=5000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.1,
        default_scaler="standard",
        exogenous_support=ExogenousSupport.FULL
    ),
    
    "AutoGRU": ModelCharacteristics(
        model_name="AutoGRU",
        base_model_name="GRU",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=5000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.1,
        default_scaler="standard",
        exogenous_support=ExogenousSupport.FULL
    ),
    
    "AutoDeepAR": ModelCharacteristics(
        model_name="AutoDeepAR",
        base_model_name="DeepAR",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=8000,
        dropout_supported=True,
        dropout_key="lstm_dropout",  # 特殊: DeepARのみ異なるキー名
        dropout_default=0.1,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Probabilistic model, outputs distribution instead of point forecast"
    ),
    
    "AutoDilatedRNN": ModelCharacteristics(
        model_name="AutoDilatedRNN",
        base_model_name="DilatedRNN",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=6000,
        dropout_supported=False,  # ドロップアウト非対応
        default_scaler="robust",
        exogenous_support=ExogenousSupport.FULL
    ),
    
    # --- CNN系 ---
    "AutoTCN": ModelCharacteristics(
        model_name="AutoTCN",
        base_model_name="TCN",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=7000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.2,
        default_scaler="standard",
        exogenous_support=ExogenousSupport.FULL
    ),
    
    "AutoBiTCN": ModelCharacteristics(
        model_name="AutoBiTCN",
        base_model_name="BiTCN",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=8000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.5,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Bidirectional TCN"
    ),
    
    "AutoTimesNet": ModelCharacteristics(
        model_name="AutoTimesNet",
        base_model_name="TimesNet",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=12000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.1,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="2D convolution over time series patches"
    ),
    
    # --- MLP/Linear系 ---
    "AutoMLP": ModelCharacteristics(
        model_name="AutoMLP",
        base_model_name="MLP",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=5000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.0,
        default_scaler="standard",
        exogenous_support=ExogenousSupport.FULL
    ),
    
    "AutoDLinear": ModelCharacteristics(
        model_name="AutoDLinear",
        base_model_name="DLinear",
        has_continuous_params=True,
        has_discrete_params=False,  # 主に連続空間
        has_categorical_params=True,
        typical_search_space_size=3000,
        dropout_supported=False,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Simple decomposition + linear, very fast"
    ),
    
    "AutoNLinear": ModelCharacteristics(
        model_name="AutoNLinear",
        base_model_name="NLinear",
        has_continuous_params=True,
        has_discrete_params=False,
        has_categorical_params=True,
        typical_search_space_size=2000,
        dropout_supported=False,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Normalized linear, extremely simple baseline"
    ),
    
    # --- NBEATS系 ---
    "AutoNBEATS": ModelCharacteristics(
        model_name="AutoNBEATS",
        base_model_name="NBEATS",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=10000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.0,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.NONE,
        notes="Neural basis expansion, interpretable stacks"
    ),
    
    "AutoNBEATSx": ModelCharacteristics(
        model_name="AutoNBEATSx",
        base_model_name="NBEATSx",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=12000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.0,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="NBEATS with exogenous variables support"
    ),
    
    "AutoNHITS": ModelCharacteristics(
        model_name="AutoNHITS",
        base_model_name="NHITS",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=10000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.0,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Hierarchical interpolation for time series"
    ),
    
    # --- その他特殊系 ---
    "AutoDeepNPTS": ModelCharacteristics(
        model_name="AutoDeepNPTS",
        base_model_name="DeepNPTS",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=8000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.1,
        default_scaler="standard",  # 特殊: standardがデフォルト
        exogenous_support=ExogenousSupport.FULL,
        notes="Non-parametric time series model"
    ),
    
    "AutoTiDE": ModelCharacteristics(
        model_name="AutoTiDE",
        base_model_name="TiDE",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=8000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.3,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Time series Dense Encoder, efficient MLP-based model"
    ),
    
    "AutoTimeMixer": ModelCharacteristics(
        model_name="AutoTimeMixer",
        base_model_name="TimeMixer",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=10000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.1,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Multivariate mixing with decomposition"
    ),
    
    "AutoRMoK": ModelCharacteristics(
        model_name="AutoRMoK",
        base_model_name="RMoK",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=9000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.0,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Recurrent Mixture of Kernels"
    ),
    
    "AutoStemGNN": ModelCharacteristics(
        model_name="AutoStemGNN",
        base_model_name="StemGNN",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=11000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.5,
        default_scaler="standard",
        exogenous_support=ExogenousSupport.FULL,
        notes="Spectral Temporal GNN, good for multivariate series"
    ),
    
    "AutoMLPMultivariate": ModelCharacteristics(
        model_name="AutoMLPMultivariate",
        base_model_name="MLPMultivariate",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=6000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.0,
        default_scaler="standard",
        exogenous_support=ExogenousSupport.FULL,
        notes="MLP for multivariate forecasting"
    ),
    
    # プレースホルダー（今後追加される可能性のあるモデル）
    "AutoTimeXer": ModelCharacteristics(
        model_name="AutoTimeXer",
        base_model_name="TimeXer",
        has_continuous_params=True,
        has_discrete_params=True,
        has_categorical_params=True,
        typical_search_space_size=10000,
        dropout_supported=True,
        dropout_key="dropout",
        dropout_default=0.1,
        default_scaler="identity",
        exogenous_support=ExogenousSupport.FULL,
        notes="Cross-series attention model"
    ),
}


def get_model_characteristics(model_name: str) -> ModelCharacteristics:
    """
    モデル名からModelCharacteristicsを取得
    
    Args:
        model_name: モデル名（例: "AutoTFT"）
        
    Returns:
        ModelCharacteristics
        
    Raises:
        ValueError: 未定義のモデル名の場合
        
    Example:
        >>> char = get_model_characteristics("AutoTFT")
        >>> print(char.dropout_key)
        "dropout"
    """
    if model_name not in MODEL_CHARACTERISTICS:
        available = ", ".join(sorted(MODEL_CHARACTERISTICS.keys()))
        raise ValueError(
            f"Unknown model: {model_name}\n"
            f"Available models: {available}"
        )
    return MODEL_CHARACTERISTICS[model_name]


def list_all_models() -> List[str]:
    """全モデル名のリストを取得"""
    return sorted(MODEL_CHARACTERISTICS.keys())


def filter_models(
    optimization_type: Optional[OptimizationType] = None,
    supports_dropout: Optional[bool] = None,
    exogenous_support: Optional[ExogenousSupport] = None
) -> List[str]:
    """
    条件に一致するモデルをフィルタリング
    
    Args:
        optimization_type: 最適化タイプ
        supports_dropout: ドロップアウト対応の有無
        exogenous_support: 外生変数サポート
        
    Returns:
        条件に一致するモデル名のリスト
        
    Example:
        >>> models = filter_models(
        ...     optimization_type=OptimizationType.CONTINUOUS,
        ...     supports_dropout=False
        ... )
        >>> print(models)
        ['AutoDLinear', 'AutoNLinear']
    """
    results = []
    for name, char in MODEL_CHARACTERISTICS.items():
        if optimization_type and char.optimization_type != optimization_type:
            continue
        if supports_dropout is not None and char.dropout_supported != supports_dropout:
            continue
        if exogenous_support and char.exogenous_support != exogenous_support:
            continue
        results.append(name)
    return sorted(results)
