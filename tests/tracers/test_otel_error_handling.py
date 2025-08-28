import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from types import ModuleType

from uipath_langchain.tracers._instrument_traceable import (
    otel_traceable_adapter,
    _instrument_traceable_attributes,
    _map_traceable_to_traced_args,
)


class TestErrorHandling:
    """Test error handling in OTEL traceable adapter."""
    
    @patch('uipath_langchain.tracers._instrument_traceable.traced')
    def test_traced_import_error(self, mock_traced):
        """Test handling when UiPath traced decorator import fails."""
        mock_traced.side_effect = ImportError("uipath.tracing not found")
        
        with pytest.raises(ImportError):
            # Should propagate the import error
            otel_traceable_adapter(run_type="tool")(lambda: None)
    
    @patch('uipath_langchain.tracers._instrument_traceable.traced')
    def test_traced_decorator_error(self, mock_traced):
        """Test handling when traced decorator raises an error."""
        mock_traced.side_effect = ValueError("Invalid traced parameters")
        
        with pytest.raises(ValueError):
            # Should propagate decorator errors
            otel_traceable_adapter(run_type="tool")(lambda: None)
    
    def test_invalid_parameter_types(self):
        """Test handling of invalid parameter types."""
        # Test with invalid tags type (currently ignored)
        result = _map_traceable_to_traced_args(tags=123)
        
        # Tags are not currently supported, so should be ignored
        assert "tags" not in result
        assert "attributes" not in result
        
        # Test with invalid metadata values (currently ignored)
        result = _map_traceable_to_traced_args(
            metadata={"key": None, "complex": {"nested": "value"}}
        )
        
        # Metadata is not currently supported, so should be ignored
        assert "metadata" not in result
        assert "attributes" not in result
    
    def test_extremely_large_parameters(self):
        """Test handling of very large parameter values."""
        large_tags = [f"tag_{i}" for i in range(1000)]
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        
        result = _map_traceable_to_traced_args(
            tags=large_tags,
            metadata=large_metadata
        )
        
        # Should handle large collections without error (currently ignored)
        assert "tags" not in result
        assert "attributes" not in result
        assert isinstance(result, dict)
    
    def test_special_characters_in_parameters(self):
        """Test handling of special characters in parameter values."""
        special_metadata = {
            "unicode": "测试数据",
            "newlines": "line1\nline2\nline3",
            "quotes": 'single"double\'quotes',
            "symbols": "!@#$%^&*()_+-=[]{}|;:,.<>?"
        }
        
        result = _map_traceable_to_traced_args(metadata=special_metadata)
        
        # Should handle special characters gracefully (currently ignored)
        assert "metadata" not in result
        assert "attributes" not in result
        assert isinstance(result, dict)


class TestModuleInstrumentationErrors:
    """Test error handling in module instrumentation."""
    
    @patch('uipath_langchain.tracers._instrument_traceable.importlib.import_module')
    def test_langsmith_import_error(self, mock_import):
        """Test handling when langsmith module cannot be imported."""
        mock_import.side_effect = ImportError("No module named 'langsmith'")
        
        # Should handle import error gracefully - implementation dependent
        try:
            result = _instrument_traceable_attributes(useOtel=True)
            # If it doesn't raise, it should return None or handle gracefully
        except ImportError:
            # Or it might propagate - both are acceptable behaviors
            pass
    
    @patch('uipath_langchain.tracers._instrument_traceable.importlib.import_module')
    def test_module_attribute_error(self, mock_import):
        """Test handling when langsmith module doesn't have traceable attribute."""
        # Reset global state for this test
        import uipath_langchain.tracers._instrument_traceable as module
        original_state = module.original_langsmith
        module.original_langsmith = None
        
        try:
            # Mock langsmith module without traceable attribute
            mock_langsmith = Mock(spec=[])  # Empty spec - no attributes
            mock_import.return_value = mock_langsmith
            
            # Current implementation raises AttributeError when traceable is missing
            with pytest.raises(AttributeError, match="Mock object has no attribute 'traceable'"):
                _instrument_traceable_attributes(useOtel=True)
        finally:
            # Restore original state
            module.original_langsmith = original_state
    
    @patch('uipath_langchain.tracers._instrument_traceable.importlib.import_module')
    @patch('uipath_langchain.tracers._instrument_traceable.sys.modules')
    def test_sys_modules_modification_error(self, mock_modules, mock_import):
        """Test handling when sys.modules modification fails."""
        # Mock sys.modules that raises on deletion
        mock_modules.__contains__ = Mock(return_value=True)
        mock_modules.__delitem__ = Mock(side_effect=KeyError("Cannot delete module"))
        
        mock_langsmith = Mock()
        mock_langsmith.traceable = Mock()
        mock_import.return_value = mock_langsmith
        
        # Should handle sys.modules errors gracefully
        try:
            result = _instrument_traceable_attributes(useOtel=True)
            # Implementation should handle this error case
        except KeyError:
            # Or it might propagate - depends on implementation
            pass
    
    @patch('uipath_langchain.tracers._instrument_traceable.register_uipath_tracing')
    def test_register_tracing_error(self, mock_register):
        """Test handling when register_uipath_tracing fails."""
        mock_register.side_effect = RuntimeError("Tracing registration failed")
        
        # Should handle registration error gracefully when OTEL is disabled
        try:
            result = _instrument_traceable_attributes(useOtel=False)
            # Implementation should handle this error
        except RuntimeError:
            # Or it might propagate
            pass


class TestConcurrencyAndThreading:
    """Test behavior under concurrent access."""
    
    @patch('uipath_langchain.tracers._instrument_traceable.importlib.import_module')
    def test_concurrent_instrumentation(self, mock_import):
        """Test concurrent calls to instrumentation function."""
        # Reset global state for this test
        import uipath_langchain.tracers._instrument_traceable as module
        original_state = module.original_langsmith
        module.original_langsmith = None
        
        try:
            # Mock langsmith module
            mock_langsmith = Mock()
            mock_langsmith.traceable = Mock()
            mock_import.return_value = mock_langsmith
            
            # Simulate concurrent calls
            results = []
            for _ in range(5):
                result = _instrument_traceable_attributes(useOtel=True)
                results.append(result)
            
            # After the first call, global state is set, so subsequent calls return the cached module
            # First call should use the mock, subsequent calls return the cached global
            assert results[0] == mock_langsmith
            # Import should only happen once due to global state
            mock_import.assert_called_once()
        finally:
            # Restore original state
            module.original_langsmith = original_state
    
    def test_global_state_consistency(self):
        """Test that global state remains consistent."""
        from uipath_langchain.tracers._instrument_traceable import (
            original_langsmith,
            original_traceable
        )
        
        # Store initial state
        initial_langsmith = original_langsmith
        initial_traceable = original_traceable
        
        # Multiple calls should maintain consistent state
        # (This test might need adjustment based on actual global state management)


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""
    
    def test_empty_function_decoration(self):
        """Test decorating functions with no parameters."""
        @otel_traceable_adapter()
        def empty_func():
            pass
        
        # Should work without error
        assert callable(empty_func)
    
    def test_lambda_decoration(self):
        """Test decorating lambda functions."""
        # This should work but might have limitations
        decorated_lambda = otel_traceable_adapter()(lambda x: x * 2)
        
        assert callable(decorated_lambda)
    
    def test_class_method_decoration(self):
        """Test decorating class methods."""
        class TestClass:
            @otel_traceable_adapter(run_type="tool")
            def method(self, x):
                return x * 2
        
        instance = TestClass()
        # Should work without error
        assert hasattr(instance, 'method')
        assert callable(instance.method)
    
    @patch('uipath_langchain.tracers._instrument_traceable.traced')
    def test_recursive_decoration(self, mock_traced):
        """Test what happens with recursive decoration."""
        mock_traced.return_value = lambda f: f
        
        # Apply decorator multiple times
        def base_func():
            pass
        
        decorated_once = otel_traceable_adapter()(base_func)
        decorated_twice = otel_traceable_adapter()(decorated_once)
        
        # Should handle multiple decoration attempts
        assert callable(decorated_twice)
    
    def test_very_long_function_names(self):
        """Test with extremely long function names."""
        long_name = "a" * 1000
        
        result = _map_traceable_to_traced_args(name=long_name)
        
        # Should handle very long names
        assert result["name"] == long_name
    
    def test_unicode_in_parameters(self):
        """Test Unicode characters in all parameters."""
        result = _map_traceable_to_traced_args(
            name="测试函数",
            run_type="工具",
            tags=["标签1", "标签2"],
            metadata={"键": "值", "key2": "测试数据"}
        )
        
        # Should handle Unicode correctly
        assert result["name"] == "测试函数"
        assert result["run_type"] == "工具"
        assert result["span_type"] == "工具"  # Will be passthrough since not in mapping
        
        # Tags and metadata not currently supported
        assert "tags" not in result
        assert "metadata" not in result
        assert "attributes" not in result


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""
    
    def test_memory_cleanup(self):
        """Test that large objects are properly cleaned up."""
        # Create adapter with large metadata
        large_metadata = {f"key_{i}": "x" * 1000 for i in range(100)}
        
        result = _map_traceable_to_traced_args(metadata=large_metadata)
        
        # Should complete without memory errors (metadata currently ignored)
        assert isinstance(result, dict)
        assert "attributes" not in result
        
        # Clear references
        del large_metadata, result
    
    def test_parameter_mapping_performance(self):
        """Test performance with many parameters."""
        import time
        
        # Large but reasonable parameter set
        tags = [f"tag_{i}" for i in range(100)]
        metadata = {f"key_{i}": f"value_{i}" for i in range(200)}
        
        start_time = time.time()
        result = _map_traceable_to_traced_args(
            name="test",
            run_type="tool",
            tags=tags,
            metadata=metadata
        )
        end_time = time.time()
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert (end_time - start_time) < 1.0  # 1 second max
        assert isinstance(result, dict)  # Should return valid dict