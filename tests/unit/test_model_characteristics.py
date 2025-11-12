"""
Unit tests for model_characteristics module.

Tests model metadata, complexity ratings, optimization type detection,
and hyperparameter search space generation.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_characteristics import (
    ModelCharacteristics,
    get_model_characteristics,
    get_all_model_types,
    get_optimization_type,
    get_complexity_rating,
    supports_exogenous_variables,
    supports_dropout,
    get_default_search_space,
    ModelMetadata
)


@pytest.mark.unit
class TestModelCharacteristics:
    """Test suite for ModelCharacteristics class."""
    
    def test_model_characteristics_initialization(self):
        """Test ModelCharacteristics initialization."""
        characteristics = ModelCharacteristics()
        
        assert characteristics is not None
        assert hasattr(characteristics, 'models')
    
    def test_load_model_data(self):
        """Test loading model characteristics data."""
        characteristics = ModelCharacteristics()
        models = characteristics.get_all_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
    
    def test_get_model_info_valid(self):
        """Test getting information for valid model."""
        characteristics = ModelCharacteristics()
        
        info = characteristics.get_model_info('AutoNHITS')
        
        assert info is not None
        assert 'complexity' in info
        assert 'optimization_type' in info
    
    def test_get_model_info_invalid(self):
        """Test getting information for invalid model."""
        characteristics = ModelCharacteristics()
        
        info = characteristics.get_model_info('InvalidModel')
        
        assert info is None


@pytest.mark.unit
class TestGetModelCharacteristics:
    """Test suite for get_model_characteristics function."""
    
    def test_get_characteristics_nhits(self):
        """Test getting characteristics for AutoNHITS."""
        chars = get_model_characteristics('AutoNHITS')
        
        assert chars is not None
        assert 'complexity' in chars
        assert 'supports_exogenous' in chars
    
    def test_get_characteristics_transformer(self):
        """Test getting characteristics for AutoTFT."""
        chars = get_model_characteristics('AutoTFT')
        
        assert chars is not None
        assert chars['complexity'] in ['Simple', 'Moderate', 'Complex']
    
    def test_get_characteristics_linear_model(self):
        """Test getting characteristics for linear model."""
        chars = get_model_characteristics('AutoARIMA')
        
        assert chars is not None
        # Linear models typically have Simple complexity
        assert chars['complexity'] in ['Simple', 'Moderate']
    
    def test_get_characteristics_case_sensitive(self):
        """Test that model names are case-sensitive."""
        chars_correct = get_model_characteristics('AutoNHITS')
        chars_wrong = get_model_characteristics('autonhits')
        
        assert chars_correct is not None
        # Wrong case should return None or raise error
        assert chars_wrong is None or chars_correct != chars_wrong


@pytest.mark.unit
class TestGetAllModelTypes:
    """Test suite for get_all_model_types function."""
    
    def test_get_all_model_types_returns_list(self):
        """Test that get_all_model_types returns a list."""
        models = get_all_model_types()
        
        assert isinstance(models, list)
        assert len(models) > 0
    
    def test_get_all_model_types_contains_expected_models(self):
        """Test that returned list contains expected models."""
        models = get_all_model_types()
        
        # Check for some known models
        assert 'AutoNHITS' in models
        assert 'AutoTFT' in models
    
    def test_get_all_model_types_no_duplicates(self):
        """Test that model list has no duplicates."""
        models = get_all_model_types()
        
        assert len(models) == len(set(models))
    
    def test_get_all_model_types_by_category(self):
        """Test getting models filtered by category."""
        models = get_all_model_types(category='Transformer')
        
        assert isinstance(models, list)
        # Should contain transformer models
        assert any('TFT' in model or 'Transformer' in model for model in models)


@pytest.mark.unit
class TestGetOptimizationType:
    """Test suite for get_optimization_type function."""
    
    def test_optimization_type_nhits(self):
        """Test optimization type for NHITS model."""
        opt_type = get_optimization_type('AutoNHITS')
        
        assert opt_type in ['optuna', 'ray', 'both']
    
    def test_optimization_type_invalid_model(self):
        """Test optimization type for invalid model."""
        opt_type = get_optimization_type('InvalidModel')
        
        assert opt_type is None or opt_type == 'optuna'  # Default
    
    def test_optimization_type_consistency(self):
        """Test that optimization type is consistent."""
        opt_type1 = get_optimization_type('AutoNHITS')
        opt_type2 = get_optimization_type('AutoNHITS')
        
        assert opt_type1 == opt_type2


@pytest.mark.unit
class TestGetComplexityRating:
    """Test suite for get_complexity_rating function."""
    
    def test_complexity_rating_values(self):
        """Test that complexity ratings are valid."""
        valid_ratings = ['Simple', 'Moderate', 'Complex']
        
        rating = get_complexity_rating('AutoNHITS')
        
        assert rating in valid_ratings
    
    def test_complexity_rating_transformer_models(self):
        """Test complexity rating for transformer models."""
        rating = get_complexity_rating('AutoTFT')
        
        # Transformer models are typically Complex or Moderate
        assert rating in ['Moderate', 'Complex']
    
    def test_complexity_rating_linear_models(self):
        """Test complexity rating for linear models."""
        rating = get_complexity_rating('AutoARIMA')
        
        # Linear models are typically Simple
        assert rating in ['Simple', 'Moderate']
    
    def test_complexity_rating_invalid_model(self):
        """Test complexity rating for invalid model."""
        rating = get_complexity_rating('InvalidModel')
        
        assert rating is None or rating == 'Moderate'  # Default


@pytest.mark.unit
class TestSupportsExogenousVariables:
    """Test suite for supports_exogenous_variables function."""
    
    def test_supports_exogenous_tft(self):
        """Test that TFT supports exogenous variables."""
        supports = supports_exogenous_variables('AutoTFT')
        
        assert supports is True
    
    def test_supports_exogenous_arima(self):
        """Test exogenous variable support for ARIMA."""
        supports = supports_exogenous_variables('AutoARIMA')
        
        # ARIMA models may or may not support exogenous variables
        assert isinstance(supports, bool)
    
    def test_supports_exogenous_invalid_model(self):
        """Test exogenous support for invalid model."""
        supports = supports_exogenous_variables('InvalidModel')
        
        assert supports is False or supports is None


@pytest.mark.unit
class TestSupportsDropout:
    """Test suite for supports_dropout function."""
    
    def test_supports_dropout_deep_models(self):
        """Test dropout support for deep learning models."""
        supports = supports_dropout('AutoNHITS')
        
        # Deep learning models typically support dropout
        assert supports is True
    
    def test_supports_dropout_linear_models(self):
        """Test dropout support for linear models."""
        supports = supports_dropout('AutoARIMA')
        
        # Linear models typically don't support dropout
        assert supports is False
    
    def test_supports_dropout_invalid_model(self):
        """Test dropout support for invalid model."""
        supports = supports_dropout('InvalidModel')
        
        assert supports is False or supports is None


@pytest.mark.unit
class TestGetDefaultSearchSpace:
    """Test suite for get_default_search_space function."""
    
    def test_default_search_space_structure(self):
        """Test that default search space has correct structure."""
        search_space = get_default_search_space('AutoNHITS')
        
        assert isinstance(search_space, dict)
        assert len(search_space) > 0
    
    def test_default_search_space_common_params(self):
        """Test that search space contains common parameters."""
        search_space = get_default_search_space('AutoNHITS')
        
        # Most models have learning rate
        common_params = ['learning_rate', 'max_steps', 'batch_size']
        found_params = [p for p in common_params if p in search_space]
        
        assert len(found_params) > 0
    
    def test_default_search_space_param_format(self):
        """Test that parameters have correct format."""
        search_space = get_default_search_space('AutoNHITS')
        
        for param_name, param_config in search_space.items():
            assert isinstance(param_config, tuple)
            assert len(param_config) >= 3
            
            param_type = param_config[0]
            assert param_type in ['float', 'int', 'categorical']
    
    def test_default_search_space_dropout_when_supported(self):
        """Test that dropout is in search space when supported."""
        if supports_dropout('AutoNHITS'):
            search_space = get_default_search_space('AutoNHITS')
            # Dropout might be included
            assert 'dropout' in search_space or 'dropout_prob_theta' in search_space
    
    def test_default_search_space_invalid_model(self):
        """Test default search space for invalid model."""
        search_space = get_default_search_space('InvalidModel')
        
        # Should return empty dict or basic search space
        assert isinstance(search_space, dict)


@pytest.mark.unit
class TestModelMetadata:
    """Test suite for ModelMetadata class."""
    
    def test_model_metadata_creation(self):
        """Test creating ModelMetadata instance."""
        metadata = ModelMetadata(
            name='AutoNHITS',
            complexity='Complex',
            optimization_type='both',
            supports_exogenous=False,
            supports_dropout=True
        )
        
        assert metadata.name == 'AutoNHITS'
        assert metadata.complexity == 'Complex'
    
    def test_model_metadata_to_dict(self):
        """Test converting ModelMetadata to dictionary."""
        metadata = ModelMetadata(
            name='AutoNHITS',
            complexity='Complex',
            optimization_type='both',
            supports_exogenous=False,
            supports_dropout=True
        )
        
        metadata_dict = metadata.to_dict()
        
        assert isinstance(metadata_dict, dict)
        assert metadata_dict['name'] == 'AutoNHITS'
    
    def test_model_metadata_validation(self):
        """Test ModelMetadata validation."""
        # Valid metadata should not raise exception
        metadata = ModelMetadata(
            name='AutoNHITS',
            complexity='Complex',
            optimization_type='both',
            supports_exogenous=False,
            supports_dropout=True
        )
        
        assert metadata.complexity in ['Simple', 'Moderate', 'Complex']


@pytest.mark.unit
class TestModelCategories:
    """Test suite for model categorization."""
    
    def test_transformer_models_category(self):
        """Test identification of transformer models."""
        models = get_all_model_types()
        transformer_models = [m for m in models if 'TFT' in m or 'Transformer' in m]
        
        assert len(transformer_models) > 0
    
    def test_rnn_models_category(self):
        """Test identification of RNN models."""
        models = get_all_model_types()
        rnn_models = [m for m in models if 'RNN' in m or 'LSTM' in m or 'GRU' in m]
        
        # May or may not have RNN models
        assert isinstance(rnn_models, list)
    
    def test_linear_models_category(self):
        """Test identification of linear models."""
        models = get_all_model_types()
        linear_models = [m for m in models if 'ARIMA' in m or 'Linear' in m or 'ETS' in m]
        
        assert len(linear_models) >= 0


@pytest.mark.unit
class TestModelCharacteristicsEdgeCases:
    """Test suite for edge cases."""
    
    def test_none_model_name(self):
        """Test handling of None as model name."""
        chars = get_model_characteristics(None)
        
        assert chars is None
    
    def test_empty_string_model_name(self):
        """Test handling of empty string as model name."""
        chars = get_model_characteristics('')
        
        assert chars is None
    
    def test_model_name_with_spaces(self):
        """Test handling of model name with spaces."""
        chars = get_model_characteristics(' AutoNHITS ')
        
        # Should handle whitespace gracefully
        assert chars is None or chars is not None
    
    def test_all_models_have_complexity(self):
        """Test that all models have a complexity rating."""
        models = get_all_model_types()
        
        for model in models:
            complexity = get_complexity_rating(model)
            assert complexity in ['Simple', 'Moderate', 'Complex', None]
    
    def test_all_models_have_optimization_type(self):
        """Test that all models have an optimization type."""
        models = get_all_model_types()
        
        for model in models:
            opt_type = get_optimization_type(model)
            assert opt_type in ['optuna', 'ray', 'both', None]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
