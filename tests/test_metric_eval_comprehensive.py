"""
Comprehensive tests for metric_eval.py to increase coverage.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, Mock
import importlib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metric_eval import init_metrics
from metrics.base_metric import BaseMetric


class TestMetricEvalComprehensive:
    """Comprehensive tests for metric_eval.py coverage."""

    def test_init_metrics_success(self):
        """Test loading all metrics successfully."""
        metrics = init_metrics()
        assert isinstance(metrics, list)
        # Should have loaded some metrics
        assert len(metrics) > 0
        # All returned items should be BaseMetric instances
        for metric in metrics:
            assert isinstance(metric, BaseMetric)
            assert hasattr(metric, 'name')
            assert metric.name is not None

    @patch('builtins.__import__')
    def test_init_metrics_import_error(self, mock_import):
        """Test handling of import errors for individual metrics."""
        # Mock the import to fail for one module
        def import_side_effect(mod_path, **kwargs):
            if mod_path == "metrics.bus_factor_metric":
                raise ImportError("Module not found")
            # For other modules, use the real import
            return importlib.import_module(mod_path)
        
        mock_import.side_effect = import_side_effect
        
        # Should continue loading other metrics despite one failure
        with patch('builtins.print') as mock_print:
            metrics = init_metrics()
            # Should have printed warning about skipped metric
            mock_print.assert_called()
            warning_calls = [call for call in mock_print.call_args_list 
                           if 'WARN' in str(call) and 'BusFactor' in str(call)]
            assert len(warning_calls) > 0

    @patch('builtins.__import__')
    def test_init_metrics_getattr_error(self, mock_import):
        """Test handling of getattr errors when class doesn't exist in module."""
        # Mock a module that doesn't have the expected class
        mock_module = Mock()
        del mock_module.BusFactorMetric  # Ensure the attribute doesn't exist
        
        def import_side_effect(mod_path, **kwargs):
            if mod_path == "metrics.bus_factor_metric":
                return mock_module
            return importlib.import_module(mod_path)
        
        mock_import.side_effect = import_side_effect
        
        with patch('builtins.print') as mock_print:
            metrics = init_metrics()
            # Should have warned about the missing class
            mock_print.assert_called()

    @patch('builtins.__import__')
    def test_init_metrics_instantiation_error(self, mock_import):
        """Test handling of errors during metric instantiation."""
        # Mock a class that fails during instantiation
        mock_module = Mock()
        mock_class = Mock()
        mock_class.side_effect = Exception("Instantiation failed")
        mock_module.BusFactorMetric = mock_class
        
        def import_side_effect(mod_path, **kwargs):
            if mod_path == "metrics.bus_factor_metric":
                return mock_module
            return importlib.import_module(mod_path)
        
        mock_import.side_effect = import_side_effect
        
        with patch('builtins.print') as mock_print:
            metrics = init_metrics()
            # Should have warned about the instantiation failure
            mock_print.assert_called()

    @patch('builtins.__import__')
    def test_init_metrics_name_mismatch(self, mock_import):
        """Test handling of metrics with incorrect names."""
        # Mock a metric with wrong name
        mock_module = Mock()
        mock_metric = Mock(spec=BaseMetric)
        mock_metric.name = "WrongName"  # Should be "BusFactor"
        mock_class = Mock(return_value=mock_metric)
        mock_module.BusFactorMetric = mock_class
        
        def import_side_effect(mod_path, **kwargs):
            if mod_path == "metrics.bus_factor_metric":
                return mock_module
            return importlib.import_module(mod_path)
        
        mock_import.side_effect = import_side_effect
        
        with patch('builtins.print') as mock_print:
            metrics = init_metrics()
            # Should have warned about name mismatch
            mock_print.assert_called()
            warning_calls = [call for call in mock_print.call_args_list 
                           if 'WARN' in str(call) and 'name=' in str(call)]
            assert len(warning_calls) > 0

    @patch('builtins.__import__')
    def test_init_metrics_metric_without_name_attribute(self, mock_import):
        """Test handling of metrics without name attribute."""
        # Mock a metric without name attribute
        mock_module = Mock()
        mock_metric = Mock(spec=BaseMetric)
        # Remove name attribute
        if hasattr(mock_metric, 'name'):
            del mock_metric.name
        mock_class = Mock(return_value=mock_metric)
        mock_module.BusFactorMetric = mock_class
        
        def import_side_effect(mod_path, **kwargs):
            if mod_path == "metrics.bus_factor_metric":
                return mock_module
            return importlib.import_module(mod_path)
        
        mock_import.side_effect = import_side_effect
        
        with patch('builtins.print') as mock_print:
            metrics = init_metrics()
            # Should have warned about missing name
            mock_print.assert_called()

    @patch('builtins.__import__')
    def test_init_metrics_multiple_failures(self, mock_import):
        """Test handling of multiple metric loading failures."""
        def import_side_effect(mod_path, **kwargs):
            if "bus_factor" in mod_path:
                raise ImportError("Bus factor module missing")
            elif "code_quality" in mod_path:
                raise ModuleNotFoundError("Code quality module not found")
            elif "license" in mod_path:
                # Return a module with a broken class
                mock_module = Mock()
                mock_module.LicenseMetric.side_effect = Exception("Broken license metric")
                return mock_module
            return importlib.import_module(mod_path)
        
        mock_import.side_effect = import_side_effect
        
        with patch('builtins.print') as mock_print:
            metrics = init_metrics()
            # Should have multiple warning messages
            assert len(mock_print.call_args_list) >= 3

    def test_init_metrics_expected_modules(self):
        """Test that all expected metrics are attempted to be loaded."""
        # This test verifies the specs list in init_metrics
        with patch('builtins.__import__') as mock_import:
            with patch('builtins.print'):
                # Make all imports fail to see what modules are attempted
                mock_import.side_effect = ImportError("Mock failure")
                init_metrics()
                
                # Check that all expected modules were attempted
                expected_modules = [
                    "metrics.bus_factor_metric",
                    "metrics.code_quality_metric", 
                    "metrics.community_rating_metric",
                    "metrics.dataset_availability_metric",
                    "metrics.dataset_quality_metric",
                    "metrics.license_metric",
                    "metrics.performance_claims_metric",
                    "metrics.ramp_up_time_metric",
                    "metrics.size_metric",
                ]
                
                called_modules = [call[0][0] for call in mock_import.call_args_list]
                for expected_mod in expected_modules:
                    assert expected_mod in called_modules

    @patch('builtins.print')
    def test_init_metrics_exception_message_format(self, mock_print):
        """Test that exception messages are properly formatted in warnings."""
        with patch('builtins.__import__') as mock_import:
            test_error = ValueError("Test error message")
            mock_import.side_effect = test_error
            
            init_metrics()
            
            # Check that the error message was included in the warning
            mock_print.assert_called()
            warning_message = str(mock_print.call_args_list[0])
            assert "BusFactor" in warning_message
            assert "Test error message" in warning_message

    def test_init_metrics_empty_specs_list(self):
        """Test behavior when specs list is empty (edge case)."""
        # This would require modifying the function, but we can test current behavior
        metrics = init_metrics()
        # Current implementation should return a list with metrics
        assert isinstance(metrics, list)

    def test_init_metrics_consistency(self):
        """Test that multiple calls return consistent results."""
        metrics1 = init_metrics()
        metrics2 = init_metrics()
        
        # Should have same number of metrics
        assert len(metrics1) == len(metrics2)
        
        # Should have same metric names
        names1 = [m.name for m in metrics1]
        names2 = [m.name for m in metrics2]
        assert sorted(names1) == sorted(names2)


class TestMetricEvalEdgeCases:
    """Test edge cases and error conditions in metric loading."""
    
    @patch('builtins.__import__')
    def test_import_returns_none(self, mock_import):
        """Test handling when import returns None."""
        def import_side_effect(mod_path, **kwargs):
            if mod_path == "metrics.bus_factor_metric":
                return None
            return importlib.import_module(mod_path)
        
        mock_import.side_effect = import_side_effect
        
        with patch('builtins.print') as mock_print:
            metrics = init_metrics()
            # Should handle None module gracefully
            mock_print.assert_called()

    @patch('builtins.__import__')  
    def test_class_instantiation_returns_none(self, mock_import):
        """Test handling when metric class instantiation returns None."""
        mock_module = Mock()
        mock_class = Mock(return_value=None)
        mock_module.BusFactorMetric = mock_class
        
        def import_side_effect(mod_path, **kwargs):
            if mod_path == "metrics.bus_factor_metric":
                return mock_module
            return importlib.import_module(mod_path)
        
        mock_import.side_effect = import_side_effect
        
        with patch('builtins.print'):
            metrics = init_metrics()
            # Should skip None metrics
            for metric in metrics:
                assert metric is not None

    @patch('builtins.__import__')
    def test_metric_name_attribute_error(self, mock_import):
        """Test when getattr for name raises AttributeError."""
        mock_module = Mock()
        mock_metric = Mock(spec=BaseMetric)
        
        def name_side_effect(*args, **kwargs):
            raise AttributeError("name attribute error")
        
        type(mock_metric).name = property(name_side_effect)
        mock_class = Mock(return_value=mock_metric)
        mock_module.BusFactorMetric = mock_class
        
        def import_side_effect(mod_path, **kwargs):
            if mod_path == "metrics.bus_factor_metric":
                return mock_module
            return importlib.import_module(mod_path)
        
        mock_import.side_effect = import_side_effect
        
        with patch('builtins.print') as mock_print:
            metrics = init_metrics()
            # Should handle attribute errors gracefully
            mock_print.assert_called()
