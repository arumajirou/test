# src/validation.py
"""
設定検証モジュール
ハイパーパラメータ設定、モデル設定、実行環境の検証を担当
"""
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """検証結果を格納するデータクラス"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    corrected_config: Optional[Dict[str, Any]] = None


class ConfigValidator:
    """設定検証クラス"""
    
    # サポートされるバックエンド
    SUPPORTED_BACKENDS = {"ray", "optuna"}
    
    # サポートされるサンプラー（Optuna）
    OPTUNA_SAMPLERS = {
        "TPESampler", "CmaEsSampler", "RandomSampler", 
        "GridSampler", "BruteForceSampler"
    }
    
    # サポートされるサーチアルゴリズム（Ray）
    RAY_SEARCH_ALGORITHMS = {
        "BasicVariantGenerator", "BayesOptSearch", "HyperOptSearch",
        "OptunaSearch", "AxSearch", "TuneBOHB"
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: Trueの場合、警告もエラーとして扱う
        """
        self.strict_mode = strict_mode
    
    def validate_backend_config(
        self, 
        backend: str,
        config: Dict[str, Any],
        num_samples: int,
        cpus: int,
        gpus: int
    ) -> ValidationResult:
        """
        バックエンド設定の検証
        
        Args:
            backend: 使用するバックエンド（'ray' or 'optuna'）
            config: ハイパーパラメータ設定
            num_samples: 試行回数
            cpus: CPU数
            gpus: GPU数
            
        Returns:
            ValidationResult: 検証結果
        """
        errors = []
        warnings = []
        corrected_config = config.copy()
        
        # バックエンドの検証
        if backend not in self.SUPPORTED_BACKENDS:
            errors.append(
                f"Unsupported backend '{backend}'. "
                f"Supported: {self.SUPPORTED_BACKENDS}"
            )
        
        # num_samplesの検証
        if num_samples < 1:
            errors.append(f"num_samples must be >= 1, got {num_samples}")
        elif num_samples > 1000:
            warnings.append(
                f"num_samples={num_samples} is very large. "
                "This may take a long time."
            )
        
        # CPUs/GPUsの検証
        if cpus < 0:
            errors.append(f"cpus must be >= 0, got {cpus}")
        if gpus < 0:
            errors.append(f"gpus must be >= 0, got {gpus}")
        
        # GPU利用可能性の検証
        if gpus > 0 and not torch.cuda.is_available():
            errors.append(
                f"gpus={gpus} specified but CUDA is not available"
            )
        
        # 利用可能なGPU数の検証
        if gpus > 0 and torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if gpus > available_gpus:
                warnings.append(
                    f"Requested {gpus} GPUs but only {available_gpus} available. "
                    f"Using {available_gpus} GPUs."
                )
                corrected_config['gpus'] = available_gpus
        
        # バックエンド固有の検証
        if backend == "ray":
            result = self._validate_ray_config(config)
        else:  # optuna
            result = self._validate_optuna_config(config)
        
        errors.extend(result.errors)
        warnings.extend(result.warnings)
        if result.corrected_config:
            corrected_config.update(result.corrected_config)
        
        is_valid = len(errors) == 0 and (not self.strict_mode or len(warnings) == 0)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            corrected_config=corrected_config if not is_valid else None
        )
    
    def _validate_ray_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Ray固有の設定検証"""
        errors = []
        warnings = []
        
        # Ray Tuneのサンプル形式検証
        from ray import tune
        
        for key, value in config.items():
            if hasattr(value, 'sampler'):
                # 有効なサンプル形式かチェック
                if not isinstance(value, (
                    tune.search.sample.Integer,
                    tune.search.sample.Float,
                    tune.search.sample.Categorical,
                    tune.search.sample.Uniform,
                    tune.search.sample.LogUniform,
                    tune.search.sample.Quantized
                )):
                    warnings.append(
                        f"Parameter '{key}' has unknown sampler type: {type(value)}"
                    )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    # def _validate_optuna_config(self, config: Dict[str, Any]) -> ValidationResult:
    #     """Optuna固有の設定検証"""
    #     errors = []
    #     warnings = []
        
    #     # Optunaでは動的な設定なので、基本的な型チェックのみ
    #     if not isinstance(config, dict):
    #         errors.append("Config must be a dictionary for Optuna backend")
        
    #     # 必須パラメータの確認
    #     required_keys = {'max_steps', 'learning_rate', 'batch_size'}
    #     missing_keys = required_keys - set(config.keys())
    #     if missing_keys:
    #         warnings.append(
    #             f"Recommended parameters missing: {missing_keys}"
    #         )
        
    #     return ValidationResult(
    #         is_valid=len(errors) == 0,
    #         errors=errors,
    #         warnings=warnings
    #     )
    
    # validation.py 内 _validate_optuna_config()
    def _validate_optuna_config(self, config: Dict[str, Any]) -> ValidationResult:
        errors, warnings = [], []
        # ここを callable 許容へ
        if not (isinstance(config, dict) or callable(config)):
            errors.append("Config for Optuna must be dict or callable(trial)->dict")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        if isinstance(config, dict):
            required = {"max_steps", "learning_rate", "batch_size"}
            missing = required - set(config.keys())
            if missing:
                warnings.append(f"Recommended parameters missing: {missing}")
        return ValidationResult(is_valid=len(errors)==0, errors=errors, warnings=warnings)

    def validate_model_config(
        self,
        model_class_name: str,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """
        モデル設定の検証
        
        Args:
            model_class_name: モデルクラス名（例: "AutoNHITS", "AutoTFT"）
            config: モデル設定
            
        Returns:
            ValidationResult: 検証結果
        """
        errors = []
        warnings = []
        corrected_config = config.copy()
        
        # 共通パラメータの検証
        if 'max_steps' in config:
            max_steps = config['max_steps']
            if isinstance(max_steps, (int, float)) and max_steps <= 0:
                errors.append(f"max_steps must be > 0, got {max_steps}")
        
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if isinstance(lr, (int, float)):
                if lr <= 0 or lr > 1:
                    warnings.append(
                        f"learning_rate={lr} is unusual. "
                        "Typical range: [1e-5, 0.1]"
                    )
        
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if isinstance(batch_size, (int, float)) and batch_size < 1:
                errors.append(f"batch_size must be >= 1, got {batch_size}")
            elif isinstance(batch_size, (int, float)) and batch_size > 1024:
                warnings.append(
                    f"batch_size={batch_size} is very large. "
                    "This may cause OOM errors."
                )
        
        # val_check_stepsの検証
        if 'val_check_steps' in config and 'max_steps' in config:
            val_steps = config['val_check_steps']
            max_steps = config['max_steps']
            if isinstance(val_steps, (int, float)) and isinstance(max_steps, (int, float)):
                if val_steps > max_steps:
                    warnings.append(
                        f"val_check_steps ({val_steps}) > max_steps ({max_steps}). "
                        "Validation will only run once at the end."
                    )
        
        # early_stop_patience_stepsの検証
        if 'early_stop_patience_steps' in config:
            patience = config['early_stop_patience_steps']
            if isinstance(patience, (int, float)) and patience < 0:
                errors.append(
                    f"early_stop_patience_steps must be >= 0, got {patience}"
                )
        
        is_valid = len(errors) == 0 and (not self.strict_mode or len(warnings) == 0)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            corrected_config=corrected_config if not is_valid else None
        )
    
    def validate_environment(self) -> ValidationResult:
        """
        実行環境の検証
        
        Returns:
            ValidationResult: 検証結果
        """
        errors = []
        warnings = []
        
        # PyTorchの検証
        try:
            import torch
            torch_version = torch.__version__
            logger.info(f"PyTorch version: {torch_version}")
        except ImportError:
            errors.append("PyTorch is not installed")
        
        # CUDA検証
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA version: {cuda_version}, GPUs: {gpu_count}")
        else:
            warnings.append("CUDA is not available. Using CPU only.")
        
        # メモリ検証
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        
        if available_gb < 4:
            warnings.append(
                f"Low available memory: {available_gb:.2f} GB. "
                "Consider reducing batch_size."
            )
        
        # ディスク容量検証
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024 ** 3)
        
        if free_gb < 10:
            warnings.append(
                f"Low disk space: {free_gb:.2f} GB. "
                "MLflow artifacts may fail to save."
            )
        
        # 必要なライブラリの検証
        required_libs = [
            'neuralforecast',
            'optuna',
            'ray',
            'mlflow',
            'lightning'
        ]
        
        for lib in required_libs:
            try:
                __import__(lib)
            except ImportError:
                errors.append(f"Required library '{lib}' is not installed")
        
        is_valid = len(errors) == 0 and (not self.strict_mode or len(warnings) == 0)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
    def validate_mlflow_config(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None
    ) -> ValidationResult:
        """
        MLflow設定の検証
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: 実験名
            
        Returns:
            ValidationResult: 検証結果
        """
        errors = []
        warnings = []
        
        # MLflowのインポート検証
        try:
            import mlflow
        except ImportError:
            errors.append("MLflow is not installed")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Tracking URIの検証
        if tracking_uri:
            if tracking_uri.startswith('http'):
                # リモートサーバーの接続確認
                try:
                    import requests
                    response = requests.get(f"{tracking_uri}/health", timeout=5)
                    if response.status_code != 200:
                        warnings.append(
                            f"MLflow server at {tracking_uri} returned status {response.status_code}"
                        )
                except Exception as e:
                    warnings.append(
                        f"Could not connect to MLflow server at {tracking_uri}: {str(e)}"
                    )
            elif tracking_uri.startswith('file://') or tracking_uri.startswith('/'):
                # ローカルパスの確認
                path = tracking_uri.replace('file://', '')
                if not os.path.exists(os.path.dirname(path)):
                    warnings.append(
                        f"MLflow tracking directory does not exist: {path}"
                    )
        
        # 実験名の検証
        if experiment_name:
            if len(experiment_name) > 256:
                errors.append(
                    f"experiment_name too long ({len(experiment_name)} chars). "
                    "Max: 256 chars"
                )
            if not experiment_name.strip():
                errors.append("experiment_name cannot be empty or whitespace only")
        
        is_valid = len(errors) == 0 and (not self.strict_mode or len(warnings) == 0)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )


class DataValidator:
    """データ検証クラス"""
    
    @staticmethod
    def validate_dataset(dataset: Any) -> ValidationResult:
        """
        データセットの検証
        
        Args:
            dataset: NeuralForecastのデータセット
            
        Returns:
            ValidationResult: 検証結果
        """
        errors = []
        warnings = []
        
        # データセットの型検証
        if dataset is None:
            errors.append("Dataset cannot be None")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # pandas DataFrameの検証
        try:
            import pandas as pd
            if not isinstance(dataset, pd.DataFrame):
                errors.append(f"Dataset must be pandas DataFrame, got {type(dataset)}")
                return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        except ImportError:
            errors.append("pandas is not installed")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # 必須カラムの検証
        required_columns = {'unique_id', 'ds', 'y'}
        missing_columns = required_columns - set(dataset.columns)
        if missing_columns:
            errors.append(
                f"Dataset missing required columns: {missing_columns}"
            )
        
        # データサイズの検証
        n_rows = len(dataset)
        if n_rows == 0:
            errors.append("Dataset is empty")
        elif n_rows < 100:
            warnings.append(
                f"Dataset is very small ({n_rows} rows). "
                "Model may not train well."
            )
        
        # 欠損値の検証
        if 'y' in dataset.columns:
            null_count = dataset['y'].isnull().sum()
            if null_count > 0:
                warnings.append(
                    f"Target variable 'y' has {null_count} missing values"
                )
        
        # 時系列の連続性検証
        if 'ds' in dataset.columns and 'unique_id' in dataset.columns:
            for uid in dataset['unique_id'].unique():
                ts_data = dataset[dataset['unique_id'] == uid].sort_values('ds')
                if len(ts_data) < 2:
                    warnings.append(
                        f"Time series '{uid}' has less than 2 observations"
                    )
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_horizon(
        h: int,
        dataset_length: int,
        input_size: Optional[int] = None
    ) -> ValidationResult:
        """
        予測ホライゾンの検証
        
        Args:
            h: 予測ホライゾン
            dataset_length: データセットの長さ
            input_size: 入力サイズ（ルックバック期間）
            
        Returns:
            ValidationResult: 検証結果
        """
        errors = []
        warnings = []
        
        if h < 1:
            errors.append(f"Horizon h must be >= 1, got {h}")
        
        if dataset_length > 0:
            ratio = h / dataset_length
            if ratio > 0.5:
                warnings.append(
                    f"Horizon ({h}) is more than 50% of dataset length ({dataset_length}). "
                    "This may lead to poor generalization."
                )
        
        if input_size and input_size > 0:
            if h > input_size * 2:
                warnings.append(
                    f"Horizon ({h}) is much larger than input_size ({input_size}). "
                    "Consider increasing input_size."
                )
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )


def validate_all(
    backend: str,
    config: Dict[str, Any],
    num_samples: int,
    cpus: int,
    gpus: int,
    model_class_name: str,
    dataset: Any,
    h: int,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: Optional[str] = None,
    strict_mode: bool = False
) -> Dict[str, ValidationResult]:
    """
    全ての検証を実行
    
    Args:
        backend: バックエンド
        config: 設定
        num_samples: 試行回数
        cpus: CPU数
        gpus: GPU数
        model_class_name: モデルクラス名
        dataset: データセット
        h: 予測ホライゾン
        mlflow_tracking_uri: MLflow tracking URI
        mlflow_experiment_name: MLflow実験名
        strict_mode: 厳格モード
        
    Returns:
        Dict[str, ValidationResult]: 各検証項目の結果
    """
    config_validator = ConfigValidator(strict_mode=strict_mode)
    data_validator = DataValidator()
    
    results = {
        'environment': config_validator.validate_environment(),
        'backend': config_validator.validate_backend_config(
            backend, config, num_samples, cpus, gpus
        ),
        'model': config_validator.validate_model_config(
            model_class_name, config
        ),
        'dataset': data_validator.validate_dataset(dataset),
        'horizon': data_validator.validate_horizon(
            h, len(dataset) if dataset is not None else 0
        )
    }
    
    if mlflow_tracking_uri or mlflow_experiment_name:
        results['mlflow'] = config_validator.validate_mlflow_config(
            mlflow_tracking_uri, mlflow_experiment_name
        )
    
    return results


def print_validation_results(results: Dict[str, ValidationResult]) -> bool:
    """
    検証結果を出力
    
    Args:
        results: 検証結果の辞書
        
    Returns:
        bool: 全ての検証が成功したかどうか
    """
    all_valid = True
    
    for category, result in results.items():
        print(f"\n{'='*60}")
        print(f"Validation: {category.upper()}")
        print(f"{'='*60}")
        
        if result.is_valid:
            print(f"✓ Status: PASSED")
        else:
            print(f"✗ Status: FAILED")
            all_valid = False
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for i, error in enumerate(result.errors, 1):
                print(f"  {i}. {error}")
        
        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for i, warning in enumerate(result.warnings, 1):
                print(f"  {i}. {warning}")
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print("✗ SOME VALIDATIONS FAILED")
    print(f"{'='*60}\n")
    
    return all_valid
