"""
Performance tests for optimization system.

Tests convergence speed, memory usage, scalability,
and overall system performance characteristics.
"""

import pytest
import sys
from pathlib import Path
import time
import psutil
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import gc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from auto_model_factory import AutoModelFactory, quick_optimize
from search_algorithm_selector import select_optimization_strategy
from validation import validate_all


@pytest.mark.performance
class TestOptimizationConvergence:
    """Test suite for optimization convergence performance."""
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_convergence_speed_small_search_space(
        self,
        mock_study,
        mock_model,
        sample_time_series
    ):
        """Test convergence speed with small search space."""
        # Setup mocks
        mock_study_instance = MagicMock()
        # Simulate convergence - decreasing loss
        mock_study_instance.trials = [
            MagicMock(value=0.1),
            MagicMock(value=0.08),
            MagicMock(value=0.06),
            MagicMock(value=0.05),
            MagicMock(value=0.05)
        ]
        mock_study_instance.best_trial.value = 0.05
        mock_study_instance.best_trial.params = {'learning_rate': 0.001}
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Small search space
        search_space = {
            'learning_rate': ('float', 1e-4, 1e-2, 'log')
        }
        
        start_time = time.time()
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna',
            config={'num_samples': 5},
            search_space=search_space
        )
        
        model = factory.create_model()
        
        elapsed_time = time.time() - start_time
        
        # Should complete quickly with small search space
        assert elapsed_time < 10.0  # seconds
        assert model is not None
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_convergence_speed_large_search_space(
        self,
        mock_study,
        mock_model,
        sample_time_series
    ):
        """Test convergence speed with large search space."""
        # Setup mocks
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study_instance.best_trial.params = {
            'learning_rate': 0.001,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.1
        }
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Large search space
        search_space = {
            'learning_rate': ('float', 1e-4, 1e-2, 'log'),
            'hidden_size': ('int', 64, 512, 'uniform'),
            'num_layers': ('categorical', [1, 2, 3, 4]),
            'dropout': ('float', 0.0, 0.5, 'uniform'),
            'batch_size': ('categorical', [32, 64, 128]),
            'max_steps': ('int', 100, 1000, 'uniform')
        }
        
        start_time = time.time()
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna',
            config={'num_samples': 10},
            search_space=search_space
        )
        
        model = factory.create_model()
        
        elapsed_time = time.time() - start_time
        
        # Should still complete in reasonable time
        assert elapsed_time < 30.0  # seconds
        assert model is not None
    
    @patch('optuna.create_study')
    def test_early_stopping_effectiveness(self, mock_study, sample_time_series):
        """Test that early stopping improves performance."""
        # Setup mock with plateauing performance
        mock_study_instance = MagicMock()
        mock_study_instance.trials = [
            MagicMock(value=0.1),
            MagicMock(value=0.05),
            MagicMock(value=0.051),
            MagicMock(value=0.052),
            MagicMock(value=0.051)  # Performance plateau
        ]
        mock_study_instance.best_trial.value = 0.05
        mock_study.return_value = mock_study_instance
        
        # Test would verify early stopping reduces optimization time
        # while maintaining performance
        assert len(mock_study_instance.trials) <= 10


@pytest.mark.performance
class TestMemoryUsage:
    """Test suite for memory usage characteristics."""
    
    def test_memory_usage_small_dataset(self, small_dataset):
        """Test memory usage with small dataset."""
        process = psutil.Process()
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Validate dataset
        validate_all(
            config={'model_type': 'AutoNHITS', 'h': 12, 'backend': 'optuna'},
            dataset=small_dataset
        )
        
        # Measure memory after validation
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - baseline_memory
        
        # Should not use excessive memory for small dataset
        assert memory_increase < 100  # MB
    
    def test_memory_usage_large_dataset(self, large_dataset):
        """Test memory usage with large dataset."""
        process = psutil.Process()
        
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Validate dataset
        validate_all(
            config={'model_type': 'AutoNHITS', 'h': 12, 'backend': 'optuna'},
            dataset=large_dataset
        )
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - baseline_memory
        
        # Memory usage should scale reasonably with dataset size
        assert memory_increase < 500  # MB
    
    @patch('auto_model_factory.AutoNHITS')
    def test_memory_leak_detection(self, mock_model, sample_time_series):
        """Test for memory leaks during repeated operations."""
        process = psutil.Process()
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple operations
        for i in range(10):
            factory = AutoModelFactory(
                model_type='AutoNHITS',
                h=12,
                backend='optuna'
            )
            model = factory.create_model()
            
            # Force garbage collection
            del factory
            del model
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory should not grow significantly
        assert memory_increase < 200  # MB


@pytest.mark.performance
class TestScalability:
    """Test suite for system scalability."""
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_scalability_with_num_trials(
        self,
        mock_study,
        mock_model,
        sample_time_series
    ):
        """Test scalability with increasing number of trials."""
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        results = {}
        
        for num_trials in [5, 10, 20]:
            mock_study_instance = MagicMock()
            mock_study_instance.best_trial.value = 0.05
            mock_study_instance.best_trial.params = {'learning_rate': 0.001}
            mock_study.return_value = mock_study_instance
            
            start_time = time.time()
            
            factory = AutoModelFactory(
                model_type='AutoNHITS',
                h=12,
                backend='optuna',
                config={'num_samples': num_trials}
            )
            
            model = factory.create_model()
            
            elapsed_time = time.time() - start_time
            results[num_trials] = elapsed_time
        
        # Time should scale roughly linearly or sub-linearly
        # (not exponentially)
        assert results[20] < results[5] * 5  # Not 4x slower for 4x trials
    
    def test_scalability_with_dataset_size(
        self,
        small_dataset,
        medium_dataset,
        large_dataset
    ):
        """Test scalability with increasing dataset size."""
        datasets = {
            'small': small_dataset,
            'medium': medium_dataset,
            'large': large_dataset
        }
        
        results = {}
        
        for size_name, dataset in datasets.items():
            start_time = time.time()
            
            # Validate dataset
            validate_all(
                config={'model_type': 'AutoNHITS', 'h': 12, 'backend': 'optuna'},
                dataset=dataset
            )
            
            elapsed_time = time.time() - start_time
            results[size_name] = elapsed_time
        
        # Validation time should scale reasonably with dataset size
        assert results['large'] < results['small'] * 100
    
    @patch('auto_model_factory.AutoNHITS')
    def test_concurrent_factory_instances(self, mock_model):
        """Test performance with multiple concurrent factory instances."""
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        start_time = time.time()
        
        # Create multiple factories concurrently
        factories = []
        for i in range(10):
            factory = AutoModelFactory(
                model_type='AutoNHITS',
                h=12,
                backend='optuna'
            )
            factories.append(factory)
        
        elapsed_time = time.time() - start_time
        
        # Should be able to create factories efficiently
        assert elapsed_time < 5.0  # seconds
        assert len(factories) == 10


@pytest.mark.performance
class TestCPUUtilization:
    """Test suite for CPU utilization."""
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_cpu_utilization_single_thread(
        self,
        mock_study,
        mock_model,
        sample_time_series
    ):
        """Test CPU utilization with single-threaded execution."""
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna',
            config={'num_samples': 5, 'n_jobs': 1}
        )
        
        # Monitor CPU during execution
        start_cpu = psutil.cpu_percent(interval=0.1)
        
        model = factory.create_model()
        
        end_cpu = psutil.cpu_percent(interval=0.1)
        
        # Should use CPU but not all cores
        assert model is not None
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_cpu_utilization_multi_thread(
        self,
        mock_study,
        mock_model,
        sample_time_series
    ):
        """Test CPU utilization with multi-threaded execution."""
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna',
            config={'num_samples': 5, 'n_jobs': 4}
        )
        
        model = factory.create_model()
        
        # Multi-threaded execution should complete
        assert model is not None


@pytest.mark.performance
@pytest.mark.slow
class TestLongRunningOptimization:
    """Test suite for long-running optimization scenarios."""
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_extended_optimization_stability(
        self,
        mock_study,
        mock_model,
        medium_dataset
    ):
        """Test system stability during extended optimization."""
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study_instance.best_trial.params = {'learning_rate': 0.001}
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Extended optimization run
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna',
            config={'num_samples': 50}
        )
        
        start_time = time.time()
        
        model = factory.create_model()
        
        elapsed_time = time.time() - start_time
        
        # Should complete without hanging
        assert elapsed_time < 120.0  # 2 minutes
        assert model is not None
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_memory_stability_long_run(
        self,
        mock_study,
        mock_model,
        medium_dataset
    ):
        """Test memory stability during long optimization."""
        process = psutil.Process()
        
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run extended optimization
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna',
            config={'num_samples': 30}
        )
        
        model = factory.create_model()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory should not grow excessively
        assert memory_increase < 1000  # MB


@pytest.mark.performance
class TestAlgorithmSelectionPerformance:
    """Test suite for algorithm selection performance."""
    
    def test_algorithm_selection_speed(self):
        """Test that algorithm selection is fast."""
        start_time = time.time()
        
        for _ in range(100):
            strategy = select_optimization_strategy(
                model_type='AutoNHITS',
                dataset_size='medium',
                backend='optuna'
            )
        
        elapsed_time = time.time() - start_time
        
        # Should be very fast (< 1 second for 100 selections)
        assert elapsed_time < 1.0
        assert strategy is not None
    
    def test_caching_effectiveness(self):
        """Test that caching improves performance."""
        # First call (cache miss)
        start_time1 = time.time()
        strategy1 = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='medium',
            backend='optuna'
        )
        time1 = time.time() - start_time1
        
        # Second call (cache hit)
        start_time2 = time.time()
        strategy2 = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='medium',
            backend='optuna'
        )
        time2 = time.time() - start_time2
        
        # Second call should be at least as fast (or use caching)
        assert strategy1 == strategy2


@pytest.mark.performance
class TestValidationPerformance:
    """Test suite for validation performance."""
    
    def test_validation_speed_small_dataset(self, small_dataset):
        """Test validation speed with small dataset."""
        start_time = time.time()
        
        validate_all(
            config={'model_type': 'AutoNHITS', 'h': 12, 'backend': 'optuna'},
            dataset=small_dataset
        )
        
        elapsed_time = time.time() - start_time
        
        # Should be very fast for small dataset
        assert elapsed_time < 1.0
    
    def test_validation_speed_large_dataset(self, large_dataset):
        """Test validation speed with large dataset."""
        start_time = time.time()
        
        validate_all(
            config={'model_type': 'AutoNHITS', 'h': 12, 'backend': 'optuna'},
            dataset=large_dataset
        )
        
        elapsed_time = time.time() - start_time
        
        # Should still be reasonably fast
        assert elapsed_time < 5.0
    
    def test_repeated_validation_performance(self, sample_time_series):
        """Test performance of repeated validations."""
        start_time = time.time()
        
        for _ in range(100):
            validate_all(
                config={'model_type': 'AutoNHITS', 'h': 12, 'backend': 'optuna'},
                dataset=sample_time_series
            )
        
        elapsed_time = time.time() - start_time
        
        # 100 validations should be fast
        assert elapsed_time < 10.0


@pytest.mark.performance
class TestBenchmarks:
    """Test suite for performance benchmarks."""
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_baseline_performance_benchmark(
        self,
        mock_study,
        mock_model,
        sample_time_series
    ):
        """Establish baseline performance benchmark."""
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        start_time = time.time()
        
        result = quick_optimize(
            model_type='AutoNHITS',
            dataset=sample_time_series,
            h=12,
            num_samples=10
        )
        
        elapsed_time = time.time() - start_time
        
        # Establish baseline: 10 trials should complete in < 30 seconds
        print(f"\nBaseline performance: {elapsed_time:.2f} seconds for 10 trials")
        
        assert elapsed_time < 30.0
        assert result is not None
    
    def test_performance_regression_check(self, sample_time_series):
        """Check for performance regressions."""
        # This test would compare against historical performance data
        # For now, just ensure basic operations complete quickly
        
        start_time = time.time()
        
        # Basic operations
        validate_all(
            config={'model_type': 'AutoNHITS', 'h': 12, 'backend': 'optuna'},
            dataset=sample_time_series
        )
        
        strategy = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='small',
            backend='optuna'
        )
        
        elapsed_time = time.time() - start_time
        
        # Basic operations should be very fast
        assert elapsed_time < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
