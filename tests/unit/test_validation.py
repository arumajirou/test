"""
Unit tests for validation module.

Tests validation of configurations, models, datasets,
and execution environments.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation import (
    ValidationError,
    ConfigValidator,
    ModelValidator,
    DatasetValidator,
    EnvironmentValidator,
    validate_all
)


@pytest.mark.unit
class TestValidationError:
    """Test suite for ValidationError exception."""
    
    def test_validation_error_creation(self):
        """Test ValidationError can be created and raised."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test error message")
        
        assert "Test error message" in str(exc_info.value)
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from Exception."""
        error = ValidationError("Test")
        assert isinstance(error, Exception)


@pytest.mark.unit
class TestConfigValidator:
    """Test suite for ConfigValidator."""
    
    def test_validate_backend_valid_optuna(self):
        """Test validation of valid Optuna backend."""
        config = {'backend': 'optuna', 'num_samples': 10}
        validator = ConfigValidator()
        
        # Should not raise exception
        validator.validate_backend(config)
    
    def test_validate_backend_valid_ray(self):
        """Test validation of valid Ray backend."""
        config = {'backend': 'ray', 'num_samples': 10}
        validator = ConfigValidator()
        
        # Should not raise exception
        validator.validate_backend(config)
    
    def test_validate_backend_invalid(self):
        """Test validation of invalid backend."""
        config = {'backend': 'invalid_backend', 'num_samples': 10}
        validator = ConfigValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_backend(config)
    
    def test_validate_num_samples_valid(self):
        """Test validation of valid num_samples."""
        config = {'num_samples': 10}
        validator = ConfigValidator()
        
        validator.validate_num_samples(config)
    
    def test_validate_num_samples_too_low(self):
        """Test validation rejects num_samples < 1."""
        config = {'num_samples': 0}
        validator = ConfigValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_num_samples(config)
    
    def test_validate_num_samples_not_integer(self):
        """Test validation rejects non-integer num_samples."""
        config = {'num_samples': 10.5}
        validator = ConfigValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_num_samples(config)
    
    def test_validate_resources_valid(self):
        """Test validation of valid resource configuration."""
        config = {'cpus': 4, 'gpus': 1}
        validator = ConfigValidator()
        
        validator.validate_resources(config)
    
    def test_validate_resources_negative_cpus(self):
        """Test validation rejects negative CPUs."""
        config = {'cpus': -1, 'gpus': 0}
        validator = ConfigValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_resources(config)
    
    def test_validate_resources_negative_gpus(self):
        """Test validation rejects negative GPUs."""
        config = {'cpus': 4, 'gpus': -1}
        validator = ConfigValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_resources(config)
    
    def test_validate_search_space_valid(self):
        """Test validation of valid search space."""
        search_space = {
            'learning_rate': ('float', 1e-4, 1e-2, 'log'),
            'hidden_size': ('int', 64, 512, 'uniform')
        }
        validator = ConfigValidator()
        
        validator.validate_search_space(search_space)
    
    def test_validate_search_space_invalid_type(self):
        """Test validation rejects invalid parameter type."""
        search_space = {
            'learning_rate': ('invalid_type', 1e-4, 1e-2, 'log')
        }
        validator = ConfigValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_search_space(search_space)
    
    def test_validate_config_complete(self, sample_config):
        """Test complete configuration validation."""
        validator = ConfigValidator()
        
        # Should validate without errors
        validator.validate_config(sample_config)


@pytest.mark.unit
class TestModelValidator:
    """Test suite for ModelValidator."""
    
    def test_validate_model_type_valid(self):
        """Test validation of valid model type."""
        validator = ModelValidator()
        
        # Should not raise exception
        validator.validate_model_type('AutoNHITS')
    
    def test_validate_model_type_invalid(self):
        """Test validation of invalid model type."""
        validator = ModelValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_model_type('InvalidModel')
    
    def test_validate_model_type_case_sensitive(self):
        """Test model type validation is case-sensitive."""
        validator = ModelValidator()
        
        # Correct case should work
        validator.validate_model_type('AutoNHITS')
        
        # Wrong case should fail
        with pytest.raises(ValidationError):
            validator.validate_model_type('autonhits')
    
    def test_validate_horizon_valid(self):
        """Test validation of valid forecast horizon."""
        validator = ModelValidator()
        
        validator.validate_horizon(12)
    
    def test_validate_horizon_too_small(self):
        """Test validation rejects horizon < 1."""
        validator = ModelValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_horizon(0)
    
    def test_validate_horizon_not_integer(self):
        """Test validation rejects non-integer horizon."""
        validator = ModelValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_horizon(12.5)
    
    def test_validate_exogenous_variables_with_support(self):
        """Test validation when model supports exogenous variables."""
        validator = ModelValidator()
        dataset = pd.DataFrame({
            'unique_id': ['A'] * 10,
            'ds': pd.date_range('2020-01-01', periods=10),
            'y': np.random.randn(10),
            'exog1': np.random.randn(10)
        })
        
        # Should not raise for models that support exogenous variables
        validator.validate_exogenous_variables('AutoTFT', dataset)
    
    def test_validate_model_config_complete(self):
        """Test complete model configuration validation."""
        validator = ModelValidator()
        config = {
            'model_type': 'AutoNHITS',
            'h': 12,
            'backend': 'optuna'
        }
        
        validator.validate_model_config(config)


@pytest.mark.unit
class TestDatasetValidator:
    """Test suite for DatasetValidator."""
    
    def test_validate_dataframe_valid(self, sample_time_series):
        """Test validation of valid DataFrame."""
        validator = DatasetValidator()
        
        validator.validate_dataframe(sample_time_series)
    
    def test_validate_dataframe_missing_columns(self):
        """Test validation rejects DataFrame with missing columns."""
        validator = DatasetValidator()
        df = pd.DataFrame({
            'unique_id': ['A'] * 10,
            'ds': pd.date_range('2020-01-01', periods=10)
            # Missing 'y' column
        })
        
        with pytest.raises(ValidationError):
            validator.validate_dataframe(df)
    
    def test_validate_dataframe_empty(self):
        """Test validation rejects empty DataFrame."""
        validator = DatasetValidator()
        df = pd.DataFrame(columns=['unique_id', 'ds', 'y'])
        
        with pytest.raises(ValidationError):
            validator.validate_dataframe(df)
    
    def test_validate_dataframe_null_values(self):
        """Test validation detects null values."""
        validator = DatasetValidator()
        df = pd.DataFrame({
            'unique_id': ['A'] * 10,
            'ds': pd.date_range('2020-01-01', periods=10),
            'y': [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        })
        
        with pytest.raises(ValidationError):
            validator.validate_dataframe(df)
    
    def test_validate_dataset_size_small(self, small_dataset):
        """Test dataset size classification - small."""
        validator = DatasetValidator()
        
        size_category = validator.get_dataset_size(small_dataset)
        assert size_category == 'small'
    
    def test_validate_dataset_size_medium(self, medium_dataset):
        """Test dataset size classification - medium."""
        validator = DatasetValidator()
        
        size_category = validator.get_dataset_size(medium_dataset)
        assert size_category == 'medium'
    
    def test_validate_dataset_size_large(self, large_dataset):
        """Test dataset size classification - large."""
        validator = DatasetValidator()
        
        size_category = validator.get_dataset_size(large_dataset)
        assert size_category == 'large'
    
    def test_validate_train_test_split_valid(self):
        """Test validation of valid train/test split."""
        validator = DatasetValidator()
        
        train_size = 80
        test_size = 20
        
        validator.validate_split_ratio(train_size, test_size)
    
    def test_validate_train_test_split_invalid_sum(self):
        """Test validation rejects split that doesn't sum to 100."""
        validator = DatasetValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_split_ratio(70, 20)
    
    def test_validate_time_series_frequency(self, sample_time_series):
        """Test validation of time series frequency."""
        validator = DatasetValidator()
        
        # Should detect daily frequency
        freq = validator.detect_frequency(sample_time_series)
        assert freq is not None


@pytest.mark.unit
class TestEnvironmentValidator:
    """Test suite for EnvironmentValidator."""
    
    def test_check_dependencies_installed(self):
        """Test checking if required dependencies are installed."""
        validator = EnvironmentValidator()
        
        # Check for common dependencies
        assert validator.check_package_installed('pandas')
        assert validator.check_package_installed('numpy')
    
    def test_check_dependencies_not_installed(self):
        """Test checking for non-existent package."""
        validator = EnvironmentValidator()
        
        assert not validator.check_package_installed('nonexistent_package_xyz')
    
    @patch('torch.cuda.is_available')
    def test_check_gpu_available_true(self, mock_cuda):
        """Test GPU availability check when GPU is available."""
        mock_cuda.return_value = True
        validator = EnvironmentValidator()
        
        assert validator.check_gpu_available()
    
    @patch('torch.cuda.is_available')
    def test_check_gpu_available_false(self, mock_cuda):
        """Test GPU availability check when GPU is not available."""
        mock_cuda.return_value = False
        validator = EnvironmentValidator()
        
        assert not validator.check_gpu_available()
    
    def test_validate_python_version(self):
        """Test Python version validation."""
        validator = EnvironmentValidator()
        
        # Should work with current Python version
        validator.validate_python_version(minimum=(3, 7))
    
    def test_check_memory_available(self):
        """Test memory availability check."""
        validator = EnvironmentValidator()
        
        available_memory = validator.get_available_memory()
        assert available_memory > 0
    
    def test_validate_environment_complete(self):
        """Test complete environment validation."""
        validator = EnvironmentValidator()
        
        # Should validate basic environment
        result = validator.validate_environment()
        assert isinstance(result, dict)


@pytest.mark.unit
class TestValidateAll:
    """Test suite for validate_all function."""
    
    def test_validate_all_success(self, sample_config, sample_time_series):
        """Test successful validation of all components."""
        result = validate_all(
            config=sample_config,
            dataset=sample_time_series
        )
        
        assert result is True or isinstance(result, dict)
    
    def test_validate_all_with_invalid_config(self, sample_time_series):
        """Test validate_all with invalid configuration."""
        invalid_config = {
            'model_type': 'InvalidModel',
            'h': 12,
            'backend': 'invalid'
        }
        
        with pytest.raises(ValidationError):
            validate_all(config=invalid_config, dataset=sample_time_series)
    
    def test_validate_all_with_invalid_dataset(self, sample_config):
        """Test validate_all with invalid dataset."""
        invalid_dataset = pd.DataFrame({
            'unique_id': ['A'] * 5,
            'ds': pd.date_range('2020-01-01', periods=5)
            # Missing 'y' column
        })
        
        with pytest.raises(ValidationError):
            validate_all(config=sample_config, dataset=invalid_dataset)


@pytest.mark.unit
class TestValidationEdgeCases:
    """Test suite for edge cases in validation."""
    
    def test_empty_config(self):
        """Test validation with empty configuration."""
        validator = ConfigValidator()
        
        with pytest.raises(ValidationError):
            validator.validate_config({})
    
    def test_none_values_in_config(self):
        """Test validation with None values."""
        validator = ConfigValidator()
        config = {'backend': None, 'num_samples': 10}
        
        with pytest.raises(ValidationError):
            validator.validate_backend(config)
    
    def test_extreme_values(self):
        """Test validation with extreme values."""
        validator = ConfigValidator()
        
        # Very large num_samples
        config = {'num_samples': 1000000}
        validator.validate_num_samples(config)
        
        # Should still validate
        assert True
    
    def test_boundary_conditions(self):
        """Test validation at boundary conditions."""
        validator = ModelValidator()
        
        # Minimum valid horizon
        validator.validate_horizon(1)
        
        # Should not raise exception
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
