"""
構造化ログ（JSON形式）の設定モジュール

このモジュールは、MLflow run_id、trial情報を含むJSON形式のログを提供し、
CloudWatch/ELK/Grafanaでの集約・分析を容易にします。

Example:
    >>> from logging_config import get_logger
    >>> logger = get_logger("my_module")
    >>> logger.info("Training started", run_id="abc123", trial_number=5)
    {"timestamp": "2025-11-12T10:30:00Z", "level": "INFO", ...}
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredJSONFormatter(logging.Formatter):
    """
    JSON形式でログを出力するカスタムFormatter
    
    Attributes:
        include_extra (bool): extra引数の内容を含めるかどうか
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """
        LogRecordをJSON文字列に変換
        
        Args:
            record: ログレコード
            
        Returns:
            JSON形式のログ文字列
        """
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }
        
        # MLflow/Optuna関連のコンテキスト情報を追加
        if hasattr(record, "run_id"):
            log_obj["mlflow_run_id"] = record.run_id
        if hasattr(record, "trial_number"):
            log_obj["trial_number"] = record.trial_number
        if hasattr(record, "model_name"):
            log_obj["model_name"] = record.model_name
        if hasattr(record, "backend"):
            log_obj["backend"] = record.backend
        
        # 例外情報を追加
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # extra引数の内容を追加
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "message", "pathname", "process", "processName",
                    "relativeCreated", "thread", "threadName", "exc_info",
                    "exc_text", "stack_info", "run_id", "trial_number",
                    "model_name", "backend"
                }:
                    log_obj[f"extra_{key}"] = value
        
        return json.dumps(log_obj, ensure_ascii=False, default=str)


class StructuredLogger:
    """
    構造化ログを提供するラッパークラス
    
    使用例:
        >>> logger = StructuredLogger("myapp")
        >>> logger.info("Process started", run_id="run_123", items_count=100)
    
    Attributes:
        logger (logging.Logger): 内部のロガーインスタンス
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_to_file: Optional[Path] = None,
        include_extra: bool = True
    ):
        """
        Args:
            name: ロガー名（通常は __name__）
            level: ログレベル
            log_to_file: ファイル出力先（Noneの場合は標準出力のみ）
            include_extra: extra引数をログに含めるか
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 既存のハンドラをクリア（重複防止）
        self.logger.handlers.clear()
        
        formatter = StructuredJSONFormatter(include_extra=include_extra)
        
        # 標準出力ハンドラ
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # ファイル出力ハンドラ（オプション）
        if log_to_file:
            log_to_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_to_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # 親ロガーへの伝播を無効化（重複防止）
        self.logger.propagate = False
    
    def _log(
        self,
        level: int,
        message: str,
        exc_info: bool = False,
        **kwargs: Any
    ) -> None:
        """
        内部ログ出力メソッド
        
        Args:
            level: ログレベル
            message: メッセージ
            exc_info: 例外情報を含めるか
            **kwargs: 追加のコンテキスト情報
        """
        extra = {k: v for k, v in kwargs.items()}
        self.logger.log(level, message, exc_info=exc_info, extra=extra)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """DEBUGレベルのログ出力"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """INFOレベルのログ出力"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """WARNINGレベルのログ出力"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(
        self,
        message: str,
        exc_info: bool = True,
        **kwargs: Any
    ) -> None:
        """ERRORレベルのログ出力（デフォルトで例外情報含む）"""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)
    
    def critical(
        self,
        message: str,
        exc_info: bool = True,
        **kwargs: Any
    ) -> None:
        """CRITICALレベルのログ出力（デフォルトで例外情報含む）"""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)


# グローバルロガーのキャッシュ
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: Optional[Path] = None,
    force_new: bool = False
) -> StructuredLogger:
    """
    構造化ロガーのファクトリー関数
    
    同じnameに対しては同一インスタンスを返す（シングルトン）。
    
    Args:
        name: ロガー名
        level: ログレベル
        log_to_file: ファイル出力先
        force_new: Trueの場合は新しいインスタンスを強制生成
        
    Returns:
        StructuredLoggerインスタンス
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Started processing", batch_size=32)
    """
    if force_new or name not in _loggers:
        _loggers[name] = StructuredLogger(
            name=name,
            level=level,
            log_to_file=log_to_file
        )
    return _loggers[name]


def configure_root_logger(level: int = logging.WARNING) -> None:
    """
    ルートロガーを設定（サードパーティライブラリのログレベル制御用）
    
    Args:
        level: ルートロガーのレベル（デフォルトはWARNING）
        
    Example:
        >>> configure_root_logger(logging.ERROR)  # 外部ライブラリのログを抑制
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 既存のハンドラをクリア
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # シンプルなコンソールハンドラを追加
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredJSONFormatter())
    root_logger.addHandler(console_handler)


# 主要ライブラリのログレベルを制限（ノイズ削減）
def suppress_noisy_loggers() -> None:
    """
    サードパーティライブラリの冗長なログを抑制
    
    対象:
    - pytorch_lightning: INFO以下を抑制
    - ray.tune: WARNING以下を抑制
    - optuna: INFO以下を抑制
    - mlflow: WARNING以下を抑制
    """
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("ray.tune").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.INFO)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)


# 初期化時に自動実行
configure_root_logger(logging.WARNING)
suppress_noisy_loggers()