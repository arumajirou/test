"""
設定ファイル管理モジュール

このモジュールは、YAMLベースの設定ファイルを読み込み、
Pythonオブジェクトとして扱うための統一インターフェースを提供します。

Example:
    >>> from config_loader import ConfigLoader
    >>> config = ConfigLoader()
    >>> mlflow_config = config.mlflow
    >>> print(mlflow_config.tracking_uri)
    'file:./mlruns'
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import os

from logging_config import get_logger

logger = get_logger(__name__)


class ConfigDict(dict):
    """
    辞書をドット記法でアクセス可能にするラッパークラス
    
    Example:
        >>> config = ConfigDict({'mlflow': {'tracking_uri': 'file:./mlruns'}})
        >>> print(config.mlflow.tracking_uri)
        'file:./mlruns'
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def get(self, key, default=None):
        """辞書のget互換メソッド"""
        try:
            return self[key]
        except KeyError:
            return default


class ConfigLoader:
    """
    設定ファイル読み込みクラス
    
    YAMLファイルから設定を読み込み、環境変数によるオーバーライドや
    設定のマージをサポートします。
    
    Attributes:
        config_dir: 設定ファイルのディレクトリ
        default_config: デフォルト設定
        model_characteristics: モデル特性定義
    
    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load_config()
        >>> print(config.mlflow.tracking_uri)
    """
    
    def __init__(
        self,
        config_dir: Union[str, Path] = "configs",
        env: Optional[str] = None
    ):
        """
        Args:
            config_dir: 設定ファイルのディレクトリパス
            env: 環境名（development, staging, production等）
                Noneの場合は環境変数ENVから取得
        """
        self.config_dir = Path(config_dir)
        self.env = env or os.getenv("ENV", "development")
        
        self.default_config: Optional[ConfigDict] = None
        self.model_characteristics: Optional[ConfigDict] = None
        
        logger.info(
            "Initialized ConfigLoader",
            config_dir=str(self.config_dir),
            environment=self.env
        )
    
    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        YAMLファイルを読み込む
        
        Args:
            file_path: YAMLファイルのパス
            
        Returns:
            読み込まれた設定の辞書
            
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            yaml.YAMLError: YAML解析エラー
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.debug(
                f"Loaded config file: {file_path.name}",
                path=str(file_path)
            )
            return config
        
        except yaml.YAMLError as e:
            logger.error(
                f"Failed to parse YAML file: {file_path}",
                exc_info=True
            )
            raise
    
    def load_config(
        self,
        config_name: str = "default_configs.yaml",
        override_config: Optional[Dict[str, Any]] = None
    ) -> ConfigDict:
        """
        設定ファイルを読み込む
        
        読み込み順序:
        1. デフォルト設定ファイル
        2. 環境別設定ファイル（存在する場合）
        3. オーバーライド設定（指定された場合）
        
        Args:
            config_name: 設定ファイル名
            override_config: 追加でオーバーライドする設定
            
        Returns:
            ConfigDict: 設定オブジェクト
        """
        # デフォルト設定を読み込み
        default_path = self.config_dir / config_name
        config = self.load_yaml(default_path)
        
        # 環境別設定を読み込み（存在する場合）
        env_config_path = self.config_dir / f"{self.env}.yaml"
        if env_config_path.exists():
            env_config = self.load_yaml(env_config_path)
            config = self.merge_configs(config, env_config)
            logger.info(
                f"Merged environment config: {self.env}",
                env_config_path=str(env_config_path)
            )
        
        # オーバーライド設定を適用
        if override_config:
            config = self.merge_configs(config, override_config)
            logger.debug("Applied override configuration")
        
        # 環境変数による設定のオーバーライド
        config = self._apply_env_vars(config)
        
        self.default_config = ConfigDict(config)
        return self.default_config
    
    def load_model_characteristics(
        self,
        config_name: str = "model_characteristics.yaml"
    ) -> ConfigDict:
        """
        モデル特性定義を読み込む
        
        Args:
            config_name: 設定ファイル名
            
        Returns:
            ConfigDict: モデル特性オブジェクト
        """
        config_path = self.config_dir / config_name
        config = self.load_yaml(config_path)
        
        self.model_characteristics = ConfigDict(config)
        logger.info("Loaded model characteristics")
        
        return self.model_characteristics
    
    def get_model_config(self, model_name: str) -> Optional[ConfigDict]:
        """
        特定のモデルの設定を取得
        
        Args:
            model_name: モデル名（例: "AutoTFT"）
            
        Returns:
            ConfigDict: モデル設定、または存在しない場合None
        """
        if self.model_characteristics is None:
            self.load_model_characteristics()
        
        models = self.model_characteristics.get('models', {})
        return models.get(model_name)
    
    def merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        設定を再帰的にマージ
        
        Args:
            base: ベース設定
            override: オーバーライド設定
            
        Returns:
            マージされた設定
        """
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result and
                isinstance(result[key], dict) and
                isinstance(value, dict)
            ):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        環境変数による設定のオーバーライド
        
        環境変数名の形式: NEURALFORECAST_<SECTION>_<KEY>
        例: NEURALFORECAST_MLFLOW_TRACKING_URI
        
        Args:
            config: 設定辞書
            
        Returns:
            環境変数を適用した設定
        """
        prefix = "NEURALFORECAST_"
        
        for env_var, value in os.environ.items():
            if not env_var.startswith(prefix):
                continue
            
            # 環境変数名を解析
            key_path = env_var[len(prefix):].lower().split('_')
            
            # 設定を更新
            current = config
            for i, key in enumerate(key_path[:-1]):
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # 型変換を試行
            final_key = key_path[-1]
            try:
                # bool値
                if value.lower() in ('true', 'false'):
                    current[final_key] = value.lower() == 'true'
                # int値
                elif value.isdigit():
                    current[final_key] = int(value)
                # float値
                elif value.replace('.', '').isdigit():
                    current[final_key] = float(value)
                # 文字列
                else:
                    current[final_key] = value
                
                logger.debug(
                    f"Applied environment variable: {env_var}",
                    value=value
                )
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Failed to apply environment variable: {env_var}",
                    error=str(e)
                )
        
        return config
    
    def save_config(
        self,
        config: Union[Dict, ConfigDict],
        output_path: Union[str, Path]
    ) -> None:
        """
        設定をYAMLファイルに保存
        
        Args:
            config: 保存する設定
            output_path: 出力ファイルパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ConfigDictを通常の辞書に変換
        if isinstance(config, ConfigDict):
            config = dict(config)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False
                )
            
            logger.info(
                f"Saved config to: {output_path}",
                path=str(output_path)
            )
        
        except Exception as e:
            logger.error(
                f"Failed to save config to: {output_path}",
                exc_info=True
            )
            raise


def load_default_config(
    config_dir: str = "configs",
    env: Optional[str] = None
) -> ConfigDict:
    """
    デフォルト設定を読み込む便利関数
    
    Args:
        config_dir: 設定ファイルのディレクトリ
        env: 環境名
        
    Returns:
        ConfigDict: 設定オブジェクト
        
    Example:
        >>> config = load_default_config()
        >>> print(config.mlflow.tracking_uri)
    """
    loader = ConfigLoader(config_dir=config_dir, env=env)
    return loader.load_config()


def load_model_characteristics(
    config_dir: str = "configs"
) -> ConfigDict:
    """
    モデル特性定義を読み込む便利関数
    
    Args:
        config_dir: 設定ファイルのディレクトリ
        
    Returns:
        ConfigDict: モデル特性オブジェクト
        
    Example:
        >>> chars = load_model_characteristics()
        >>> tft_config = chars.models.AutoTFT
        >>> print(tft_config.dropout_default)
    """
    loader = ConfigLoader(config_dir=config_dir)
    return loader.load_model_characteristics()


def get_model_config(
    model_name: str,
    config_dir: str = "configs"
) -> Optional[ConfigDict]:
    """
    特定のモデルの設定を取得する便利関数
    
    Args:
        model_name: モデル名
        config_dir: 設定ファイルのディレクトリ
        
    Returns:
        ConfigDict: モデル設定
        
    Example:
        >>> config = get_model_config("AutoTFT")
        >>> print(config.typical_search_space_size)
    """
    loader = ConfigLoader(config_dir=config_dir)
    return loader.get_model_config(model_name)


# エクスポート
__all__ = [
    'ConfigLoader',
    'ConfigDict',
    'load_default_config',
    'load_model_characteristics',
    'get_model_config'
]
