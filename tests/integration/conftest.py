"""
Pytest configuration and shared fixtures for the test suite.

This module provides common fixtures and configuration for unit, integration,
and performance tests.
"""

import sys
from pathlib import Path
from typing import Generator
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_time_series() -> pd.DataFrame:
    """
    Generate a sample time series dataset for testing.
    
    Returns:
        DataFrame with columns: unique_id, ds, y
    """
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = {
        'unique_id': ['series_1'] * 100,
        'ds': dates,
        'y': np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.1, 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_time_series_with_exog() -> pd.DataFrame:
    """
    Generate a sample time series with exogenous variables.
    
    Returns:
        DataFrame with columns: unique_id, ds, y, exog1, exog2
    """
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = {
        'unique_id': ['series_1'] * 100,
        'ds': dates,
        'y': np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.1, 100),
        'exog1': np.random.normal(0, 1, 100),
        'exog2': np.random.uniform(0, 10, 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_multivariate_time_series() -> pd.DataFrame:
    """
    Generate multiple time series for testing.
    
    Returns:
        DataFrame with multiple unique_ids
    """
    dfs = []
    for i in range(3):
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = {
            'unique_id': [f'series_{i+1}'] * 100,
            'ds': dates,
            'y': np.sin(np.linspace(0, 4 * np.pi, 100) + i) + np.random.normal(0, 0.1, 100)
        }
        dfs.append(pd.DataFrame(data))
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def small_dataset() -> pd.DataFrame:
    """Small dataset (< 1000 rows) for testing."""
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = {
        'unique_id': ['series_1'] * 500,
        'ds': dates,
        'y': np.random.normal(100, 10, 500)
    }
    return pd.DataFrame(data)


@pytest.fixture
def medium_dataset() -> pd.DataFrame:
    """Medium dataset (1000-10000 rows) for testing."""
    dates = pd.date_range(start='2020-01-01', periods=5000, freq='H')
    data = {
        'unique_id': ['series_1'] * 5000,
        'ds': dates,
        'y': np.random.normal(100, 10, 5000)
    }
    return pd.DataFrame(data)


@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Large dataset (> 10000 rows) for testing."""
    dates = pd.date_range(start='2020-01-01', periods=15000, freq='H')
    data = {
        'unique_id': ['series_1'] * 15000,
        'ds': dates,
        'y': np.random.normal(100, 10, 15000)
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for testing."""
    mock_client = MagicMock()
    mock_client.create_experiment.return_value = "test_exp_id"
    mock_client.create_run.return_value.info.run_id = "test_run_id"
    return mock_client


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration for testing."""
    return {
        'model_type': 'AutoNHITS',
        'h': 12,
        'backend': 'optuna',
        'num_samples': 10,
        'gpus': 0,
        'cpus': 2
    }


@pytest.fixture
def optuna_config() -> dict:
    """Optuna-specific configuration."""
    return {
        'backend': 'optuna',
        'num_samples': 20,
        'config': {
            'n_jobs': 2,
            'timeout': 60,
            'show_progress_bar': False
        }
    }


@pytest.fixture
def ray_config() -> dict:
    """Ray Tune-specific configuration."""
    return {
        'backend': 'ray',
        'num_samples': 20,
        'config': {
            'num_cpus': 2,
            'num_gpus': 0
        }
    }


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests."""
    import logging
    # Clear all handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.root.setLevel(logging.WARNING)
    yield
    # Cleanup after test
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


@pytest.fixture
def mock_optuna_study():
    """Mock Optuna study for testing."""
    study = MagicMock()
    study.best_trial.value = 0.05
    study.best_trial.params = {
        'learning_rate': 0.001,
        'hidden_size': 128,
        'num_layers': 2
    }
    study.trials = [MagicMock(value=0.1), MagicMock(value=0.05)]
    return study


@pytest.fixture
def sample_search_space() -> dict:
    """Sample hyperparameter search space."""
    return {
        'learning_rate': ('float', 1e-4, 1e-2, 'log'),
        'hidden_size': ('int', 64, 512, 'uniform'),
        'num_layers': ('categorical', [1, 2, 3, 4]),
        'dropout': ('float', 0.0, 0.5, 'uniform')
    }


# Performance test markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
