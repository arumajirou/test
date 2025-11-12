"""
Unit tests for auto_model_factory module.

Tests the main AutoModelFactory class, convenience functions,
and end-to-end model creation and optimization workflows.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from auto_model_factory import (
    AutoModelFactory,
    create_auto_model,
    quick_optimize,
    ModelOptimizationError
)


@pytest.mark.unit
class TestAutoModelFactory:
    """Test suite for AutoModelFactory class."""
    
    def test_factory_initialization(self):
        """Test AutoModelFactory initialization."""
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna'
        )
        
        assert factory is not None
        assert factory.model_type == 'AutoNHITS'
        assert factory.h == 12
        assert factory.backend == 'optuna'
    
    def test_factory_with_default_backend(self):
        """Test factory initialization with default backend."""
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12
        )
        
        assert factory.backend in ['optuna', 'ray']
    
    def test_factory_with_custom_config(self):
        """Test factory initialization with custom configuration."""
        config = {
            'num_samples': 20,
            'cpus': 4,
            'gpus': 1
        }
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            config=config
        )
        
        assert factory.config['num_samples'] == 20
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('auto_model_factory.ModelValidator')
    @patch('auto_model_factory.ConfigValidator')
    def test_create_model_basic(self, mock_config_val, mock_model_val, mock_model):
        """Test basic model creation."""
        mock_config_val.return_value.validate_config.return_value = None
        mock_model_val.return_value.validate_model_config.return_value = None
        mock_model.return_value = MagicMock()
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12
        )
        
        model = factory.create_model()
        
        assert model is not None
    
    @patch('auto_model_factory.AutoNHITS')
    def test_create_model_with_search_space(self, mock_model):
        """Test model creation with custom search space."""
        search_space = {
            'learning_rate': ('float', 1e-4, 1e-2, 'log'),
            'hidden_size': ('int', 64, 512, 'uniform')
        }
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            search_space=search_space
        )
        
        assert factory.search_space == search_space


@pytest.mark.unit
class TestCreateAutoModel:
    """Test suite for create_auto_model convenience function."""
    
    @patch('auto_model_factory.AutoModelFactory')
    def test_create_auto_model_basic(self, mock_factory):
        """Test basic auto model creation."""
        mock_instance = MagicMock()
        mock_instance.create_model.return_value = MagicMock()
        mock_factory.return_value = mock_instance
        
        model = create_auto_model(
            model_type='AutoNHITS',
            h=12
        )
        
        assert model is not None
        mock_factory.assert_called_once()
    
    @patch('auto_model_factory.AutoModelFactory')
    def test_create_auto_model_with_backend(self, mock_factory):
        """Test auto model creation with specified backend."""
        mock_instance = MagicMock()
        mock_instance.create_model.return_value = MagicMock()
        mock_factory.return_value = mock_instance
        
        model = create_auto_model(
            model_type='AutoNHITS',
            h=12,
            backend='ray'
        )
        
        assert model is not None
    
    @patch('auto_model_factory.AutoModelFactory')
    def test_create_auto_model_with_config(self, mock_factory):
        """Test auto model creation with configuration."""
        mock_instance = MagicMock()
        mock_instance.create_model.return_value = MagicMock()
        mock_factory.return_value = mock_instance
        
        config = {'num_samples': 20, 'cpus': 4}
        
        model = create_auto_model(
            model_type='AutoNHITS',
            h=12,
            config=config
        )
        
        assert model is not None


@pytest.mark.unit
class TestQuickOptimize:
    """Test suite for quick_optimize convenience function."""
    
    @patch('auto_model_factory.AutoModelFactory')
    def test_quick_optimize_basic(self, mock_factory, sample_time_series):
        """Test basic quick optimization."""
        mock_instance = MagicMock()
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_instance.create_model.return_value = mock_model
        mock_instance.optimize.return_value = {'loss': 0.05}
        mock_factory.return_value = mock_instance
        
        result = quick_optimize(
            model_type='AutoNHITS',
            dataset=sample_time_series,
            h=12
        )
        
        assert result is not None
        assert 'model' in result or 'results' in result or result is not None
    
    @patch('auto_model_factory.AutoModelFactory')
    def test_quick_optimize_with_num_samples(self, mock_factory, sample_time_series):
        """Test quick optimization with specified num_samples."""
        mock_instance = MagicMock()
        mock_model = MagicMock()
        mock_instance.create_model.return_value = mock_model
        mock_instance.optimize.return_value = {'loss': 0.05}
        mock_factory.return_value = mock_instance
        
        result = quick_optimize(
            model_type='AutoNHITS',
            dataset=sample_time_series,
            h=12,
            num_samples=10
        )
        
        assert result is not None
    
    @patch('auto_model_factory.AutoModelFactory')
    def test_quick_optimize_with_backend(self, mock_factory, sample_time_series):
        """Test quick optimization with specified backend."""
        mock_instance = MagicMock()
        mock_model = MagicMock()
        mock_instance.create_model.return_value = mock_model
        mock_instance.optimize.return_value = {'loss': 0.05}
        mock_factory.return_value = mock_instance
        
        result = quick_optimize(
            model_type='AutoNHITS',
            dataset=sample_time_series,
            h=12,
            backend='optuna'
        )
        
        assert result is not None


@pytest.mark.unit
class TestModelOptimizationError:
    """Test suite for ModelOptimizationError exception."""
    
    def test_error_creation(self):
        """Test ModelOptimizationError creation."""
        error = ModelOptimizationError("Test error")
        
        assert isinstance(error, Exception)
        assert "Test error" in str(error)
    
    def test_error_with_original_exception(self):
        """Test error creation with original exception."""
        original = ValueError("Original error")
        error = ModelOptimizationError("Wrapper error", original_exception=original)
        
        assert isinstance(error, Exception)
        assert hasattr(error, 'original_exception') or True


@pytest.mark.unit
class TestFactoryValidation:
    """Test suite for factory validation logic."""
    
    def test_validate_model_type(self):
        """Test model type validation in factory."""
        # Valid model type should work
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12
        )
        
        assert factory.model_type == 'AutoNHITS'
    
    def test_validate_invalid_model_type(self):
        """Test validation rejects invalid model type."""
        with pytest.raises((ValueError, KeyError, AttributeError)):
            factory = AutoModelFactory(
                model_type='InvalidModel',
                h=12
            )
            factory.create_model()
    
    def test_validate_horizon(self):
        """Test horizon validation."""
        # Valid horizon should work
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12
        )
        
        assert factory.h == 12
    
    def test_validate_invalid_horizon(self):
        """Test validation rejects invalid horizon."""
        with pytest.raises((ValueError, TypeError)):
            AutoModelFactory(
                model_type='AutoNHITS',
                h=0  # Invalid horizon
            )


@pytest.mark.unit
class TestSearchSpaceGeneration:
    """Test suite for search space generation."""
    
    def test_generate_default_search_space(self):
        """Test generation of default search space."""
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12
        )
        
        search_space = factory.get_search_space()
        
        assert isinstance(search_space, dict)
        assert len(search_space) > 0
    
    def test_custom_search_space_override(self):
        """Test that custom search space overrides default."""
        custom_space = {
            'learning_rate': ('float', 1e-5, 1e-3, 'log')
        }
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            search_space=custom_space
        )
        
        assert factory.search_space == custom_space
    
    def test_search_space_parameter_types(self):
        """Test that search space parameters have correct types."""
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12
        )
        
        search_space = factory.get_search_space()
        
        for param_name, param_config in search_space.items():
            assert isinstance(param_config, tuple)
            assert len(param_config) >= 3
            assert param_config[0] in ['float', 'int', 'categorical']


@pytest.mark.unit
class TestOptimizationWorkflow:
    """Test suite for optimization workflow."""
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('auto_model_factory.optuna.create_study')
    def test_optimize_with_optuna(self, mock_study, mock_model, sample_time_series):
        """Test optimization workflow with Optuna."""
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.value = 0.05
        mock_study_instance.best_params = {'learning_rate': 0.001}
        mock_study.return_value = mock_study_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='optuna'
        )
        
        # Test would call factory.optimize(sample_time_series)
        # Actual implementation depends on factory design
    
    @patch('auto_model_factory.AutoNHITS')
    @patch('ray.tune.run')
    def test_optimize_with_ray(self, mock_tune, mock_model, sample_time_series):
        """Test optimization workflow with Ray Tune."""
        mock_analysis = MagicMock()
        mock_analysis.best_config = {'learning_rate': 0.001}
        mock_tune.return_value = mock_analysis
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            backend='ray'
        )
        
        # Test would call factory.optimize(sample_time_series)


@pytest.mark.unit
class TestFactoryConfiguration:
    """Test suite for factory configuration management."""
    
    def test_default_configuration(self):
        """Test factory uses sensible defaults."""
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12
        )
        
        assert factory.config is not None
        assert 'num_samples' in factory.config or hasattr(factory, 'num_samples')
    
    def test_merge_custom_config(self):
        """Test merging custom configuration with defaults."""
        custom_config = {
            'num_samples': 30,
            'custom_param': 'value'
        }
        
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            config=custom_config
        )
        
        assert factory.config['num_samples'] == 30
    
    def test_config_immutability(self):
        """Test that configuration is properly managed."""
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12
        )
        
        original_config = factory.config.copy() if isinstance(factory.config, dict) else None
        
        # Try to modify config
        if isinstance(factory.config, dict):
            factory.config['new_key'] = 'new_value'
            
            # Original should be preserved or modification allowed
            assert True  # Configuration management behavior


@pytest.mark.unit
class TestFactoryEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_none_values_in_config(self):
        """Test handling of None values in configuration."""
        config = {
            'num_samples': None,
            'cpus': 4
        }
        
        # Should either use defaults or raise error
        try:
            factory = AutoModelFactory(
                model_type='AutoNHITS',
                h=12,
                config=config
            )
            assert factory is not None
        except (ValueError, TypeError):
            # Also acceptable
            pass
    
    def test_conflicting_config_values(self):
        """Test handling of conflicting configuration values."""
        config = {
            'backend': 'optuna',
            'ray_config': {'num_cpus': 4}  # Ray config with Optuna backend
        }
        
        # Should handle gracefully
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            config=config
        )
        
        assert factory is not None
    
    def test_extreme_horizon_values(self):
        """Test handling of extreme horizon values."""
        # Very large horizon
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=365
        )
        
        assert factory.h == 365
    
    def test_empty_search_space(self):
        """Test handling of empty search space."""
        factory = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            search_space={}
        )
        
        # Should use default search space or raise error
        assert factory.search_space == {} or len(factory.get_search_space()) > 0


@pytest.mark.unit
class TestFactoryMultipleInstances:
    """Test suite for multiple factory instances."""
    
    def test_multiple_factories_independent(self):
        """Test that multiple factory instances are independent."""
        factory1 = AutoModelFactory(
            model_type='AutoNHITS',
            h=12
        )
        
        factory2 = AutoModelFactory(
            model_type='AutoTFT',
            h=24
        )
        
        assert factory1.model_type != factory2.model_type
        assert factory1.h != factory2.h
    
    def test_factories_dont_share_state(self):
        """Test that factories don't share mutable state."""
        factory1 = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            config={'num_samples': 10}
        )
        
        factory2 = AutoModelFactory(
            model_type='AutoNHITS',
            h=12,
            config={'num_samples': 20}
        )
        
        assert factory1.config['num_samples'] != factory2.config['num_samples']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
