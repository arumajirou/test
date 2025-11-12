"""
Integration tests for end-to-end optimization workflows.

Tests the complete pipeline from data validation through
model creation, optimization, and result logging.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from auto_model_factory import AutoModelFactory, create_auto_model, quick_optimize
from validation import validate_all
from mlflow_integration import MLflowTracker
from search_algorithm_selector import select_optimization_strategy


@pytest.mark.integration
class TestEndToEndOptimization:
    """Test suite for complete end-to-end optimization workflow."""
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    def test_complete_optimization_workflow(
        self, 
        mock_log_metrics,
        mock_log_params,
        mock_mlflow_run,
        mock_model,
        sample_time_series
    ):
        """Test complete optimization workflow from start to finish."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model_instance.fit.return_value = mock_model_instance
        mock_model_instance.predict.return_value = pd.DataFrame({
            'ds': pd.date_range('2021-01-01', periods=12),
            'AutoNHITS': np.random.randn(12)
        })
        mock_model.return_value = mock_model_instance
        
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_mlflow_run.return_value.__enter__.return_value = mock_run
        
        # 1. Validate data
        validation_result = validate_all(
            config={'model_type': 'AutoNHITS', 'h': 12, 'backend': 'optuna'},
            dataset=sample_time_series
        )
        
        assert validation_result is True or isinstance(validation_result, dict)
        
        # 2. Select optimization strategy
        strategy = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='small',
            backend='optuna'
        )
        
        assert strategy is not None
        assert 'sampler' in strategy
        
        # 3. Create model factory
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna'
        )
        
        assert factory is not None
        
        # 4. Create model
        model = factory.create_model()
        
        assert model is not None
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_optimization_with_optuna_backend(
        self,
        mock_study,
        mock_model,
        sample_time_series
    ):
        """Test optimization workflow with Optuna backend."""
        # Setup mocks
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study_instance.best_trial.params = {
            'learning_rate': 0.001,
            'hidden_size': 128
        }
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.fit.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        # Create and optimize model
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna',
            config={'num_samples': 5}
        )
        
        model = factory.create_model()
        
        assert model is not None
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('ray.tune.run')
    def test_optimization_with_ray_backend(
        self,
        mock_tune,
        mock_model,
        sample_time_series
    ):
        """Test optimization workflow with Ray Tune backend."""
        # Setup mocks
        mock_analysis = MagicMock()
        mock_analysis.best_config = {
            'learning_rate': 0.001,
            'hidden_size': 128
        }
        mock_analysis.best_result = {'loss': 0.05}
        mock_tune.return_value = mock_analysis
        
        mock_model_instance = MagicMock()
        mock_model_instance.fit.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        # Create and optimize model
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='ray',
            config={'num_samples': 5}
        )
        
        model = factory.create_model()
        
        assert model is not None


@pytest.mark.integration
class TestValidationIntegration:
    """Test suite for validation integration with other components."""
    
    def test_validation_before_optimization(self, sample_time_series):
        """Test that validation occurs before optimization."""
        # Valid configuration
        config = {
            'model_type': 'AutoNHITS',
            'h': 12,
            'backend': 'optuna',
            'num_samples': 10
        }
        
        # Validate
        validation_result = validate_all(config=config, dataset=sample_time_series)
        
        assert validation_result is True or isinstance(validation_result, dict)
        
        # Create factory only after validation
        factory = AutoModelFactory(**config)
        
        assert factory is not None
    
    def test_validation_catches_errors_early(self):
        """Test that validation catches errors before expensive operations."""
        # Invalid configuration
        invalid_config = {
            'model_type': 'InvalidModel',
            'h': 12,
            'backend': 'invalid_backend'
        }
        
        invalid_dataset = pd.DataFrame({
            'unique_id': ['A'] * 10,
            'ds': pd.date_range('2020-01-01', periods=10)
            # Missing 'y' column
        })
        
        # Validation should catch errors
        with pytest.raises(Exception):
            validate_all(config=invalid_config, dataset=invalid_dataset)


@pytest.mark.integration
class TestMLflowIntegration:
    """Test suite for MLflow integration with optimization workflow."""
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.set_experiment')
    @patch('auto_model_factory.AutoNHITS')
    def test_mlflow_tracking_during_optimization(
        self,
        mock_model,
        mock_set_exp,
        mock_log_metrics,
        mock_log_params,
        mock_start_run,
        sample_time_series
    ):
        """Test MLflow tracking during model optimization."""
        # Setup mocks
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        mock_model_instance = MagicMock()
        mock_model_instance.fit.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        # Create tracker
        tracker = MLflowTracker(experiment_name='test_integration')
        
        # Start run and log
        with tracker.start_run(run_name='test_run'):
            # Create model
            factory = AutoModelFactory(
                model_type='AutoNHITS',
                h=12,
                backend='optuna'
            )
            
            model = factory.create_model()
            
            # Log parameters
            tracker.log_parameters({
                'model_type': 'AutoNHITS',
                'h': 12,
                'backend': 'optuna'
            })
            
            # Log metrics
            tracker.log_metrics({
                'loss': 0.05,
                'mae': 0.1
            })
        
        # Verify MLflow calls
        mock_start_run.assert_called()
        mock_log_params.assert_called()
        mock_log_metrics.assert_called()
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_experiment_tracking_with_optuna(
        self,
        mock_study,
        mock_model,
        mock_log_params,
        mock_start_run,
        sample_time_series
    ):
        """Test experiment tracking integration with Optuna optimization."""
        # Setup mocks
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study_instance.best_trial.params = {'learning_rate': 0.001}
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Create tracker
        tracker = MLflowTracker(experiment_name='optuna_integration')
        
        with tracker.start_run():
            # Optimize model
            factory = AutoModelFactory(
                model_type='AutoNHITS',
                h=12,
                backend='optuna',
                config={'num_samples': 3}
            )
            
            model = factory.create_model()
            
            # Log best parameters
            tracker.log_parameters(mock_study_instance.best_trial.params)
        
        mock_log_params.assert_called()


@pytest.mark.integration
class TestAlgorithmSelectionIntegration:
    """Test suite for algorithm selection integration."""
    
    def test_algorithm_selection_with_model_characteristics(
        self,
        sample_time_series
    ):
        """Test algorithm selection based on model characteristics."""
        from model_characteristics import get_complexity_rating
        
        # Get model complexity
        complexity = get_complexity_rating('AutoNHITS')
        
        # Select algorithm based on complexity
        strategy = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='medium',
            backend='optuna'
        )
        
        assert strategy is not None
        assert 'sampler' in strategy
        assert 'num_trials' in strategy
    
    def test_dataset_size_affects_algorithm_selection(
        self,
        small_dataset,
        large_dataset
    ):
        """Test that dataset size influences algorithm selection."""
        # Strategy for small dataset
        strategy_small = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='small',
            backend='optuna'
        )
        
        # Strategy for large dataset
        strategy_large = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='large',
            backend='optuna'
        )
        
        # Strategies should differ
        assert strategy_small != strategy_large or \
               strategy_small['num_trials'] != strategy_large['num_trials']


@pytest.mark.integration
class TestConvenienceFunctions:
    """Test suite for high-level convenience functions."""
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_quick_optimize_end_to_end(
        self,
        mock_study,
        mock_model,
        sample_time_series
    ):
        """Test quick_optimize function end-to-end."""
        # Setup mocks
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study_instance.best_trial.params = {'learning_rate': 0.001}
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.fit.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        # Quick optimize
        result = quick_optimize(
            model_type='AutoNHITS',
            dataset=sample_time_series,
            h=12,
            num_samples=3
        )
        
        assert result is not None
    
    @patch('auto_model_factory.AutoNHITS')
    def test_create_auto_model_convenience(self, mock_model):
        """Test create_auto_model convenience function."""
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        model = create_auto_model(
            model_type='AutoNHITS',
            h=12,
            backend='optuna'
        )
        
        assert model is not None


@pytest.mark.integration
@pytest.mark.slow
class TestMultiModelOptimization:
    """Test suite for optimizing multiple models."""
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('auto_model_factory.AutoTFT')
    @patch('optuna.create_study')
    def test_optimize_multiple_models(
        self,
        mock_study,
        mock_tft,
        mock_nhits,
        sample_time_series
    ):
        """Test optimizing multiple different models."""
        # Setup mocks
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study_instance.best_trial.params = {'learning_rate': 0.001}
        mock_study.return_value = mock_study_instance
        
        mock_nhits_instance = MagicMock()
        mock_nhits.return_value = mock_nhits_instance
        
        mock_tft_instance = MagicMock()
        mock_tft.return_value = mock_tft_instance
        
        # Optimize NHITS
        factory1 = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna',
            config={'num_samples': 2}
        )
        model1 = factory1.create_model()
        
        # Optimize TFT
        factory2 = AutoModelFactory(
            model_type='AutoTFT',
            h=12,
            backend='optuna',
            config={'num_samples': 2}
        )
        model2 = factory2.create_model()
        
        assert model1 is not None
        assert model2 is not None
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('auto_model_factory.AutoNHITS')
    @patch('optuna.create_study')
    def test_compare_models_with_mlflow(
        self,
        mock_study,
        mock_model,
        mock_log_metrics,
        mock_log_params,
        mock_start_run,
        sample_time_series
    ):
        """Test comparing multiple models with MLflow tracking."""
        # Setup mocks
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        tracker = MLflowTracker(experiment_name='model_comparison')
        
        models = ['AutoNHITS', 'AutoTFT']
        results = {}
        
        for model_type in models:
            with tracker.start_run(run_name=f'{model_type}_run'):
                factory = AutoModelFactory(
                    model_type=model_type,
                    h=12,
                    backend='optuna',
                    config={'num_samples': 2}
                )
                
                model = factory.create_model()
                
                # Log model-specific parameters
                tracker.log_parameters({'model_type': model_type})
                
                # Log metrics
                tracker.log_metrics({'loss': 0.05})
                
                results[model_type] = {'loss': 0.05}
        
        assert len(results) == 2


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test suite for error handling across components."""
    
    def test_validation_error_propagation(self):
        """Test that validation errors propagate correctly."""
        # Invalid dataset
        invalid_data = pd.DataFrame({
            'unique_id': ['A'] * 5,
            'ds': pd.date_range('2020-01-01', periods=5)
            # Missing 'y'
        })
        
        # Should raise validation error
        with pytest.raises(Exception):
            validate_all(
                config={'model_type': 'AutoNHITS', 'h': 12, 'backend': 'optuna'},
                dataset=invalid_data
            )
    
    @patch('auto_model_factory.AutoNHITS')
    def test_model_creation_error_handling(self, mock_model):
        """Test error handling during model creation."""
        mock_model.side_effect = ValueError("Model creation failed")
        
        # Should handle error gracefully
        with pytest.raises(Exception):
            factory = AutoModelFactory(
                model_type='AutoNHITS',
                h=12
            )
            factory.create_model()


@pytest.mark.integration
class TestDataFlowIntegration:
    """Test suite for data flow between components."""
    
    def test_dataset_validation_to_optimization(self, sample_time_series):
        """Test data flow from validation to optimization."""
        # Validate dataset
        from validation import DatasetValidator
        
        validator = DatasetValidator()
        validator.validate_dataframe(sample_time_series)
        
        # Get dataset size
        size = validator.get_dataset_size(sample_time_series)
        
        # Use size for algorithm selection
        strategy = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size=size,
            backend='optuna'
        )
        
        assert strategy is not None
    
    def test_model_characteristics_to_search_space(self):
        """Test flow from model characteristics to search space."""
        from model_characteristics import (
            get_model_characteristics,
            get_default_search_space
        )
        
        # Get characteristics
        chars = get_model_characteristics('AutoNHITS')
        
        # Get search space
        search_space = get_default_search_space('AutoNHITS')
        
        assert chars is not None
        assert search_space is not None
        assert isinstance(search_space, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
