"""
MLflow統合モジュール

このモジュールは、NeuralForecast AutoModelsの最適化プロセスを
MLflowで完全にトラッキングするための統一インターフェースを提供します。

主要機能:
- ハイパーパラメータのログ
- 中間・最終メトリクスのログ
- モデルアーティファクトの保存
- モデルレジストリへの登録
- 実験の比較・可視化

Example:
    >>> tracker = MLflowTracker("file:./mlruns", "auto_tft_experiment")
    >>> with tracker.start_run("trial_001"):
    ...     tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
    ...     tracker.log_metrics({"train_loss": 0.5, "valid_loss": 0.6}, step=10)
    ...     tracker.log_model(model, "best_model")
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import tempfile
from contextlib import contextmanager
import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    MLflow統合の統一インターフェース
    
    このクラスは、実験のライフサイクル全体を管理します：
    - 実験の作成・取得
    - runの開始・終了
    - パラメータ・メトリクス・アーティファクトのログ
    - モデルの保存・登録
    
    Attributes:
        tracking_uri (str): MLflow Tracking ServerのURI
        experiment_name (str): 実験名
        client (MlflowClient): MLflow APIクライアント
        experiment_id (str): 実験ID
        run (mlflow.ActiveRun): 現在アクティブなrun
    """
    
    def __init__(
        self,
        tracking_uri: str = "file:./mlruns",
        experiment_name: str = "neuralforecast_auto",
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            tracking_uri: MLflow Tracking ServerのURI
                         - ローカル: "file:./mlruns"
                         - リモート: "http://mlflow-server:5000"
                         - Databricks: "databricks"
            experiment_name: 実験名（同名の実験が存在する場合は再利用）
            tags: 実験レベルのタグ（例: {"project": "forecasting", "team": "ml"}）
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # MLflowクライアントの初期化
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        
        # --- 実験の作成または取得（例外に依存せず直列で実施） ---
        exp = self.client.get_experiment_by_name(experiment_name)
        if exp is None:
            # ない場合は新規作成（MLflow 2.16 では tags 引数が使用可能）
            self.experiment_id = self.client.create_experiment(experiment_name, tags=tags or {})
            logger.info(
                f"Created new MLflow experiment: {experiment_name}",
                experiment_id=self.experiment_id,
                tracking_uri=tracking_uri
            )
        else:
            self.experiment_id = exp.experiment_id
            logger.info(
                f"Using existing MLflow experiment: {experiment_name}",
                experiment_id=self.experiment_id
            )
        
        # 互換のために名前でもセットしておく（実際のrun開始はexperiment_idで行う）
        mlflow.set_experiment(experiment_name=experiment_name)
        self.run: Optional[mlflow.ActiveRun] = None
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """
        新しいrunを開始（コンテキストマネージャー）
        
        Args:
            run_name: run名（Noneの場合は自動生成）
            nested: 親runの中でネストされたrunを作成するか
            tags: runレベルのタグ
            description: runの説明
            
        Yields:
            mlflow.ActiveRun: アクティブなrun
            
        Example:
            >>> with tracker.start_run("trial_001", tags={"backend": "optuna"}):
            ...     tracker.log_params({"lr": 0.001})
            ...     # runは自動的に終了
        """
        self.run = None
        try:
            # experiment_id を明示してグローバル状態に依存しない
            self.run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested,
                tags=tags,
                description=description
            )
            logger.info(
                "Started MLflow run",
                run_id=self.run.info.run_id,
                run_name=run_name or "auto"
            )
            yield self.run
        except Exception as e:
            logger.error(
                "Error during MLflow run",
                run_id=(self.run.info.run_id if self.run else "unknown"),
                exc_info=True
            )
            if self.run:
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error_message", str(e))
            raise
        finally:
            if self.run:
                mlflow.end_run()
                logger.info("Ended MLflow run", run_id=self.run.info.run_id)
                self.run = None
    
    def log_params(
        self,
        params: Dict[str, Any],
        prefix: str = ""
    ) -> None:
        """
        ハイパーパラメータをログ
        
        MLflowはパラメータ値を文字列として保存します。
        長い値（>500文字）は自動的に切り詰められます。
        
        Args:
            params: パラメータ辞書
            prefix: パラメータ名のプレフィックス（例: "model."）
        """
        if not self.run:
            logger.warning("No active run. Call start_run() first.")
            return
        
        params_str = {}
        for key, value in params.items():
            full_key = f"{prefix}{key}" if prefix else key
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + "..."
            params_str[full_key] = str_value
        
        try:
            mlflow.log_params(params_str)
            logger.debug(
                f"Logged {len(params_str)} parameters",
                run_id=self.run.info.run_id,
                params_count=len(params_str)
            )
        except Exception:
            logger.error("Failed to log parameters", run_id=self.run.info.run_id, exc_info=True)
            raise
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        メトリクスをログ（中間値・最終値）
        """
        if not self.run:
            logger.warning("No active run. Call start_run() first.")
            return
        
        metrics_prefixed = {}
        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, float):
                if np.isnan(value):
                    logger.warning(f"Metric '{full_key}' is NaN at step {step}", run_id=self.run.info.run_id)
                    continue
                if np.isinf(value):
                    logger.warning(f"Metric '{full_key}' is Inf at step {step}", run_id=self.run.info.run_id)
                    continue
            metrics_prefixed[full_key] = float(value)
        
        try:
            mlflow.log_metrics(metrics_prefixed, step=step)
            logger.debug(
                f"Logged {len(metrics_prefixed)} metrics",
                run_id=self.run.info.run_id,
                step=step,
                metrics_count=len(metrics_prefixed)
            )
        except Exception:
            logger.error("Failed to log metrics", run_id=self.run.info.run_id, exc_info=True)
            raise
    
    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None
    ) -> None:
        """
        アーティファクト（ファイル・ディレクトリ）をログ
        """
        if not self.run:
            logger.warning("No active run. Call start_run() first.")
            return
        
        local_path = Path(local_path)
        try:
            if local_path.is_file():
                mlflow.log_artifact(str(local_path), artifact_path)
            elif local_path.is_dir():
                mlflow.log_artifacts(str(local_path), artifact_path)
            else:
                raise ValueError(f"Path does not exist: {local_path}")
            
            logger.debug(
                f"Logged artifact: {local_path.name}",
                run_id=self.run.info.run_id,
                artifact_path=artifact_path or "root"
            )
        except Exception:
            logger.error(f"Failed to log artifact: {local_path}", run_id=self.run.info.run_id, exc_info=True)
            raise
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        signature: Optional[Any] = None,
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        PyTorchモデルをMLflow形式で保存
        """
        if not self.run:
            logger.warning("No active run. Call start_run() first.")
            return
        
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                signature=signature,
                registered_model_name=registered_model_name
            )
            logger.info(
                "Logged PyTorch model",
                run_id=self.run.info.run_id,
                artifact_path=artifact_path,
                registered_name=registered_model_name
            )
        except Exception:
            logger.error("Failed to log model", run_id=self.run.info.run_id, exc_info=True)
            raise
    
    def log_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """
        DataFrameをCSVとしてログ
        """
        if not self.run:
            logger.warning("No active run. Call start_run() first.")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / filename
            df.to_csv(tmp_path, index=False)
            self.log_artifact(tmp_path, artifact_path)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        runにタグを設定
        """
        if not self.run:
            logger.warning("No active run. Call start_run() first.")
            return
        mlflow.set_tags(tags)
        logger.debug(f"Set {len(tags)} tags", run_id=self.run.info.run_id)
    
    def get_best_run(
        self,
        metric_name: str = "valid_loss",
        ascending: bool = True,
        filter_string: Optional[str] = None
    ) -> Optional[mlflow.entities.Run]:
        """
        実験内のベストrunを取得
        """
        order_by = [f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=order_by,
            max_results=1
        )
        if runs.empty:
            logger.warning("No runs found matching criteria")
            return None
        run_id = runs.iloc[0]["run_id"]
        return self.client.get_run(run_id)
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        複数のrunを比較
        """
        rows: List[Dict[str, Any]] = []
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                row = {"run_id": run_id, "run_name": run.info.run_name}
                if metrics:
                    for m in metrics:
                        row[f"metrics.{m}"] = run.data.metrics.get(m)
                else:
                    row.update({f"metrics.{k}": v for k, v in run.data.metrics.items()})
                row.update({f"params.{k}": v for k, v in run.data.params.items()})
                rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to fetch run {run_id}: {e}")
        return pd.DataFrame(rows)


def create_mlflow_callback_optuna(tracker: MLflowTracker):
    """
    Optuna用のMLflowコールバックを作成
    """
    def callback(study, trial):
        """Optuna trialの終了時にMLflowへログ"""
        with tracker.start_run(
            run_name=f"trial_{trial.number}",
            tags={
                "backend": "optuna",
                "trial_number": str(trial.number),
                "trial_state": trial.state.name
            }
        ):
            # パラメータ
            tracker.log_params(trial.params)
            # メトリクス
            if trial.value is not None:
                tracker.log_metrics({"objective_value": trial.value})
            # 中間値
            for step, value in trial.intermediate_values.items():
                tracker.log_metrics({"intermediate_value": value}, step=step)
            # 属性
            tracker.log_params({f"attr_{k}": v for k, v in trial.user_attrs.items()})
    return callback
