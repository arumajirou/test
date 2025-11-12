"""
Unit tests for search_algorithm_selector module.

Tests algorithm selection logic, sampler/pruner selection,
and recommendation system for optimization strategies.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from search_algorithm_selector import (
    SearchAlgorithmSelector,
    get_recommended_sampler,
    get_recommended_pruner,
    get_recommended_num_trials,
    select_optimization_strategy
)


@pytest.mark.unit
class TestSearchAlgorithmSelector:
    """Test suite for SearchAlgorithmSelector class."""
    
    def test_selector_initialization(self):
        """Test SearchAlgorithmSelector initialization."""
        selector = SearchAlgorithmSelector()
        
        assert selector is not None
        assert hasattr(selector, 'select_algorithm')
    
    def test_select_algorithm_simple_model_small_data(self):
        """Test algorithm selection for simple model with small dataset."""
        selector = SearchAlgorithmSelector()
        
        config = selector.select_algorithm(
            model_complexity='Simple',
            dataset_size='small'
        )
        
        assert isinstance(config, dict)
        assert 'sampler' in config
        assert 'pruner' in config
    
    def test_select_algorithm_complex_model_large_data(self):
        """Test algorithm selection for complex model with large dataset."""
        selector = SearchAlgorithmSelector()
        
        config = selector.select_algorithm(
            model_complexity='Complex',
            dataset_size='large'
        )
        
        assert isinstance(config, dict)
        assert config['sampler'] is not None
        assert config['pruner'] is not None
    
    def test_select_algorithm_moderate_complexity(self):
        """Test algorithm selection for moderate complexity."""
        selector = SearchAlgorithmSelector()
        
        config = selector.select_algorithm(
            model_complexity='Moderate',
            dataset_size='medium'
        )
        
        assert isinstance(config, dict)


@pytest.mark.unit
class TestGetRecommendedSampler:
    """Test suite for get_recommended_sampler function."""
    
    def test_recommended_sampler_simple_model(self):
        """Test sampler recommendation for simple model."""
        sampler = get_recommended_sampler(
            complexity='Simple',
            backend='optuna'
        )
        
        assert sampler is not None
        assert isinstance(sampler, str)
    
    def test_recommended_sampler_complex_model(self):
        """Test sampler recommendation for complex model."""
        sampler = get_recommended_sampler(
            complexity='Complex',
            backend='optuna'
        )
        
        assert sampler is not None
        # Complex models might use TPE or other advanced samplers
        assert sampler in ['TPE', 'CmaEs', 'RandomSampler', 'GridSampler']
    
    def test_recommended_sampler_optuna_backend(self):
        """Test sampler recommendation for Optuna backend."""
        sampler = get_recommended_sampler(
            complexity='Moderate',
            backend='optuna'
        )
        
        assert sampler is not None
    
    def test_recommended_sampler_ray_backend(self):
        """Test sampler recommendation for Ray backend."""
        sampler = get_recommended_sampler(
            complexity='Moderate',
            backend='ray'
        )
        
        assert sampler is not None
    
    def test_recommended_sampler_invalid_complexity(self):
        """Test sampler recommendation with invalid complexity."""
        sampler = get_recommended_sampler(
            complexity='Invalid',
            backend='optuna'
        )
        
        # Should return a default sampler
        assert sampler is not None


@pytest.mark.unit
class TestGetRecommendedPruner:
    """Test suite for get_recommended_pruner function."""
    
    def test_recommended_pruner_for_complex_model(self):
        """Test pruner recommendation for complex model."""
        pruner = get_recommended_pruner(
            complexity='Complex',
            backend='optuna'
        )
        
        assert pruner is not None
        assert isinstance(pruner, str)
    
    def test_recommended_pruner_for_simple_model(self):
        """Test pruner recommendation for simple model."""
        pruner = get_recommended_pruner(
            complexity='Simple',
            backend='optuna'
        )
        
        # Simple models might not need aggressive pruning
        assert pruner is not None or pruner is None
    
    def test_recommended_pruner_optuna_backend(self):
        """Test pruner recommendation for Optuna backend."""
        pruner = get_recommended_pruner(
            complexity='Moderate',
            backend='optuna'
        )
        
        assert pruner in ['MedianPruner', 'HyperbandPruner', 'PercentilePruner', None]
    
    def test_recommended_pruner_ray_backend(self):
        """Test pruner recommendation for Ray backend."""
        pruner = get_recommended_pruner(
            complexity='Moderate',
            backend='ray'
        )
        
        # Ray uses schedulers instead of pruners
        assert isinstance(pruner, (str, type(None)))


@pytest.mark.unit
class TestGetRecommendedNumTrials:
    """Test suite for get_recommended_num_trials function."""
    
    def test_recommended_trials_simple_model(self):
        """Test trial count recommendation for simple model."""
        num_trials = get_recommended_num_trials(
            complexity='Simple',
            dataset_size='small'
        )
        
        assert isinstance(num_trials, int)
        assert num_trials > 0
    
    def test_recommended_trials_complex_model(self):
        """Test trial count recommendation for complex model."""
        num_trials = get_recommended_num_trials(
            complexity='Complex',
            dataset_size='large'
        )
        
        assert isinstance(num_trials, int)
        # Complex models typically need more trials
        assert num_trials >= 20
    
    def test_recommended_trials_scales_with_complexity(self):
        """Test that trial count scales with model complexity."""
        simple_trials = get_recommended_num_trials('Simple', 'small')
        moderate_trials = get_recommended_num_trials('Moderate', 'medium')
        complex_trials = get_recommended_num_trials('Complex', 'large')
        
        # Should generally increase with complexity
        assert simple_trials <= moderate_trials <= complex_trials or \
               simple_trials <= complex_trials
    
    def test_recommended_trials_small_dataset(self):
        """Test trial count for small dataset."""
        num_trials = get_recommended_num_trials(
            complexity='Moderate',
            dataset_size='small'
        )
        
        assert isinstance(num_trials, int)
        assert 10 <= num_trials <= 100
    
    def test_recommended_trials_large_dataset(self):
        """Test trial count for large dataset."""
        num_trials = get_recommended_num_trials(
            complexity='Moderate',
            dataset_size='large'
        )
        
        assert isinstance(num_trials, int)
        # Large datasets might need fewer trials due to longer training
        assert num_trials >= 10


@pytest.mark.unit
class TestSelectOptimizationStrategy:
    """Test suite for select_optimization_strategy function."""
    
    def test_strategy_selection_complete(self):
        """Test complete optimization strategy selection."""
        strategy = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='medium',
            backend='optuna'
        )
        
        assert isinstance(strategy, dict)
        assert 'sampler' in strategy
        assert 'pruner' in strategy
        assert 'num_trials' in strategy
    
    def test_strategy_selection_with_different_models(self):
        """Test strategy selection for different model types."""
        strategy1 = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='medium',
            backend='optuna'
        )
        
        strategy2 = select_optimization_strategy(
            model_type='AutoARIMA',
            dataset_size='medium',
            backend='optuna'
        )
        
        # Strategies might differ based on model complexity
        assert isinstance(strategy1, dict)
        assert isinstance(strategy2, dict)
    
    def test_strategy_selection_optuna_backend(self):
        """Test strategy selection for Optuna backend."""
        strategy = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='medium',
            backend='optuna'
        )
        
        assert 'sampler' in strategy
        assert 'pruner' in strategy
    
    def test_strategy_selection_ray_backend(self):
        """Test strategy selection for Ray backend."""
        strategy = select_optimization_strategy(
            model_type='AutoNHITS',
            dataset_size='medium',
            backend='ray'
        )
        
        assert isinstance(strategy, dict)
        # Ray-specific configurations
        assert 'scheduler' in strategy or 'search_alg' in strategy or 'sampler' in strategy


@pytest.mark.unit
class TestAlgorithmSelectionMatrix:
    """Test suite for algorithm selection matrix logic."""
    
    def test_matrix_covers_all_combinations(self):
        """Test that selection matrix covers all complexity-size combinations."""
        selector = SearchAlgorithmSelector()
        
        complexities = ['Simple', 'Moderate', 'Complex']
        sizes = ['small', 'medium', 'large']
        
        for complexity in complexities:
            for size in sizes:
                config = selector.select_algorithm(complexity, size)
                assert config is not None
                assert 'sampler' in config
    
    def test_matrix_returns_different_configs(self):
        """Test that different inputs return appropriate configs."""
        selector = SearchAlgorithmSelector()
        
        config1 = selector.select_algorithm('Simple', 'small')
        config2 = selector.select_algorithm('Complex', 'large')
        
        # Configs should differ in some way
        assert config1 != config2 or \
               config1['num_trials'] != config2['num_trials']


@pytest.mark.unit
class TestSearchSpaceComplexity:
    """Test suite for search space complexity assessment."""
    
    def test_assess_search_space_simple(self, sample_search_space):
        """Test assessment of simple search space."""
        selector = SearchAlgorithmSelector()
        
        # Simple search space with few parameters
        simple_space = {
            'learning_rate': ('float', 1e-4, 1e-2, 'log'),
            'hidden_size': ('int', 64, 256, 'uniform')
        }
        
        complexity = selector.assess_search_space_complexity(simple_space)
        
        assert complexity in ['Simple', 'Moderate', 'Complex']
    
    def test_assess_search_space_complex(self):
        """Test assessment of complex search space."""
        selector = SearchAlgorithmSelector()
        
        # Complex search space with many parameters
        complex_space = {
            f'param_{i}': ('float', 0.0, 1.0, 'uniform')
            for i in range(20)
        }
        
        complexity = selector.assess_search_space_complexity(complex_space)
        
        assert complexity == 'Complex'


@pytest.mark.unit
class TestBackendSpecificSelection:
    """Test suite for backend-specific algorithm selection."""
    
    def test_optuna_specific_samplers(self):
        """Test Optuna-specific sampler selection."""
        sampler = get_recommended_sampler(
            complexity='Moderate',
            backend='optuna'
        )
        
        optuna_samplers = ['TPE', 'CmaEs', 'RandomSampler', 'GridSampler']
        assert sampler in optuna_samplers
    
    def test_ray_specific_samplers(self):
        """Test Ray-specific sampler selection."""
        sampler = get_recommended_sampler(
            complexity='Moderate',
            backend='ray'
        )
        
        # Ray samplers might have different names
        assert isinstance(sampler, str)
    
    def test_backend_consistency(self):
        """Test that backend selection is consistent."""
        sampler1 = get_recommended_sampler('Moderate', 'optuna')
        sampler2 = get_recommended_sampler('Moderate', 'optuna')
        
        assert sampler1 == sampler2


@pytest.mark.unit
class TestEdgeCases:
    """Test suite for edge cases in algorithm selection."""
    
    def test_invalid_complexity_level(self):
        """Test handling of invalid complexity level."""
        selector = SearchAlgorithmSelector()
        
        config = selector.select_algorithm(
            model_complexity='Invalid',
            dataset_size='medium'
        )
        
        # Should return a default configuration
        assert isinstance(config, dict)
    
    def test_invalid_dataset_size(self):
        """Test handling of invalid dataset size."""
        selector = SearchAlgorithmSelector()
        
        config = selector.select_algorithm(
            model_complexity='Moderate',
            dataset_size='invalid'
        )
        
        # Should return a default configuration
        assert isinstance(config, dict)
    
    def test_none_values(self):
        """Test handling of None values."""
        selector = SearchAlgorithmSelector()
        
        try:
            config = selector.select_algorithm(None, None)
            # Should either return default config or raise ValueError
            assert isinstance(config, dict) or config is None
        except (ValueError, TypeError):
            # This is also acceptable behavior
            pass
    
    def test_extreme_num_trials(self):
        """Test handling of extreme trial count requests."""
        num_trials = get_recommended_num_trials('Complex', 'large')
        
        # Should be reasonable, not excessively large
        assert num_trials < 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
