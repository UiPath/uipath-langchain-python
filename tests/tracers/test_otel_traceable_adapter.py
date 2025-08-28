import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from uipath_langchain.tracers._instrument_traceable import (
    _map_traceable_to_traced_args,
    otel_traceable_adapter,
    _instrument_traceable_attributes,
)


class TestMapTraceableToTracedArgs:
    """Test parameter mapping from LangSmith @traceable to UiPath @traced()."""
    
    def test_basic_mapping(self):
        """Test basic parameter mapping."""
        result = _map_traceable_to_traced_args(
            name="test_func",
            run_type="tool"
        )
        
        assert result["name"] == "test_func"
        assert result["run_type"] == "tool"
        assert result["span_type"] == "tool_call"
    
    def test_run_type_mapping(self):
        """Test run_type to span_type mapping."""
        test_cases = [
            ("tool", "tool_call"),
            ("chain", "chain_execution"),
            ("llm", "llm_call"),
            ("retriever", "retrieval"),
            ("embedding", "embedding"),
            ("prompt", "prompt_template"),
            ("parser", "output_parser"),
            ("custom_type", "custom_type"),  # passthrough
        ]
        
        for run_type, expected_span_type in test_cases:
            result = _map_traceable_to_traced_args(run_type=run_type)
            assert result["span_type"] == expected_span_type
            assert result["run_type"] == run_type
    
    def test_tags_mapping(self):
        """Test tags mapping (currently not supported by UiPath @traced)."""
        result = _map_traceable_to_traced_args(
            tags=["search", "web", "test"]
        )
        
        # Tags are currently not mapped since UiPath @traced doesn't support them
        assert "tags" not in result
        
    def test_tags_string_mapping(self):
        """Test single tag as string (currently not supported)."""
        result = _map_traceable_to_traced_args(
            tags="single_tag"
        )
        
        # Tags are currently not mapped
        assert "tags" not in result
    
    def test_metadata_mapping(self):
        """Test metadata mapping (currently not supported by UiPath @traced)."""
        metadata = {
            "version": "1.0",
            "model": "gpt-4",
            "temperature": 0.7
        }
        
        result = _map_traceable_to_traced_args(metadata=metadata)
        
        # Metadata is currently not mapped since UiPath @traced doesn't support custom attributes
        assert "metadata" not in result
        assert "attributes" not in result
    
    def test_complete_mapping(self):
        """Test complete parameter mapping."""
        result = _map_traceable_to_traced_args(
            name="research_tool",
            run_type="tool",
            tags=["research", "web"],
            metadata={"version": "2.0", "provider": "tavily"}
        )
        
        assert result["name"] == "research_tool"
        assert result["run_type"] == "tool"
        assert result["span_type"] == "tool_call"
        
        # Tags and metadata are not currently supported by UiPath @traced
        assert "tags" not in result
        assert "metadata" not in result
        assert "attributes" not in result
    
    def test_empty_parameters(self):
        """Test handling of empty/None parameters."""
        result = _map_traceable_to_traced_args()
        
        # Should return empty dict when no parameters provided
        assert result == {}
    
    def test_kwargs_ignored(self):
        """Test that extra kwargs are ignored."""
        result = _map_traceable_to_traced_args(
            name="test",
            unknown_param="ignored",
            another_param=123
        )
        
        assert result["name"] == "test"
        assert "unknown_param" not in result
        assert "another_param" not in result


class TestOtelTraceableAdapter:
    """Test the OTEL traceable adapter decorator."""
    
    @patch('uipath_langchain.tracers._instrument_traceable.traced')
    def test_decorator_without_parentheses(self, mock_traced):
        """Test @otel_traceable_adapter usage (direct decoration)."""
        mock_traced.return_value = lambda f: f  # Mock traced decorator
        
        def sample_func():
            return "test"
        
        # Apply decorator directly
        decorated_func = otel_traceable_adapter(sample_func)
        
        # Should call traced() with empty args and return decorated function
        mock_traced.assert_called_once_with()
        assert callable(decorated_func)
    
    @patch('uipath_langchain.tracers._instrument_traceable.traced')
    def test_decorator_with_parentheses(self, mock_traced):
        """Test @otel_traceable_adapter(...) usage (parameterized decoration)."""
        mock_traced_instance = Mock()
        mock_traced.return_value = mock_traced_instance
        mock_traced_instance.return_value = lambda f: f
        
        def sample_func():
            return "test"
        
        # Create parameterized decorator
        decorator = otel_traceable_adapter(
            run_type="tool",
            name="test_tool",
            tags=["test"],
            metadata={"version": "1.0"}
        )
        
        # Apply to function
        decorated_func = decorator(sample_func)
        
        # Should call traced with mapped parameters
        expected_args = {
            "name": "test_tool",
            "run_type": "tool",
            "span_type": "tool_call"
        }
        
        mock_traced.assert_called_once_with(**expected_args)
        mock_traced_instance.assert_called_once_with(sample_func)
        assert callable(decorated_func)
    
    @patch('uipath_langchain.tracers._instrument_traceable.traced')
    def test_parameter_mapping_integration(self, mock_traced):
        """Test that parameters are correctly mapped through the adapter."""
        mock_traced.return_value = lambda f: f
        
        decorator = otel_traceable_adapter(
            run_type="chain",
            name="research_chain",
            tags=["research", "web"],
            metadata={"model": "gpt-4", "temperature": 0.7}
        )
        
        def sample_func():
            pass
        
        decorator(sample_func)
        
        # Verify traced was called with correctly mapped parameters
        call_args = mock_traced.call_args[1]  # Get keyword arguments
        
        assert call_args["name"] == "research_chain"
        assert call_args["run_type"] == "chain"
        assert call_args["span_type"] == "chain_execution"
        
        # Tags and metadata are not currently supported
        assert "tags" not in call_args
        assert "metadata" not in call_args
        assert "attributes" not in call_args


class TestInstrumentTraceableAttributes:
    """Test the instrumentation function with OTEL support."""
    
    @patch('uipath_langchain.tracers._instrument_traceable.importlib.import_module')
    def test_otel_enabled(self, mock_import):
        """Test instrumentation with OTEL enabled."""
        # Reset global state for this test
        import uipath_langchain.tracers._instrument_traceable as module
        original_state = module.original_langsmith
        module.original_langsmith = None
        
        try:
            # Mock langsmith module
            mock_langsmith = Mock()
            mock_langsmith.traceable = Mock()
            mock_import.return_value = mock_langsmith
            
            # Call with OTEL enabled
            result = _instrument_traceable_attributes(useOtel=True)
            
            # Should import langsmith and replace traceable with OTEL adapter
            mock_import.assert_called_once_with("langsmith")
            assert result == mock_langsmith
            
            # traceable should be replaced with otel_traceable_adapter
            from uipath_langchain.tracers._instrument_traceable import otel_traceable_adapter
            assert mock_langsmith.traceable == otel_traceable_adapter
        finally:
            # Restore original state
            module.original_langsmith = original_state
    
    @patch('uipath_langchain.tracers._instrument_traceable.importlib.import_module')
    @patch('uipath_langchain.tracers._instrument_traceable.register_uipath_tracing')
    def test_otel_disabled(self, mock_register, mock_import):
        """Test instrumentation with OTEL disabled."""
        # Reset global state for this test
        import uipath_langchain.tracers._instrument_traceable as module
        original_state = module.original_langsmith
        module.original_langsmith = None
        
        try:
            # Mock langsmith module
            mock_langsmith = Mock()
            mock_langsmith.traceable = Mock()
            mock_import.return_value = mock_langsmith
            
            # Call with OTEL disabled
            result = _instrument_traceable_attributes(useOtel=False)
            
            # Should register custom tracing and use patched_traceable
            mock_register.assert_called_once()
            mock_import.assert_called_once_with("langsmith")
            assert result == mock_langsmith
            
            # traceable should be replaced with patched_traceable
            from uipath_langchain.tracers._instrument_traceable import patched_traceable
            assert mock_langsmith.traceable == patched_traceable
        finally:
            # Restore original state
            module.original_langsmith = original_state
    
    @patch('uipath_langchain.tracers._instrument_traceable.importlib.import_module')
    def test_existing_module_handling(self, mock_import):
        """Test handling when langsmith module is already in sys.modules."""
        # Reset global state for this test
        import uipath_langchain.tracers._instrument_traceable as module
        original_state = module.original_langsmith
        module.original_langsmith = None
        
        try:
            # Mock existing langsmith in sys.modules
            existing_langsmith = Mock()
            existing_langsmith.traceable = Mock()
            
            import sys
            with patch.dict(sys.modules, {'langsmith': existing_langsmith}):
                # Mock fresh import
                fresh_langsmith = Mock()
                fresh_langsmith.traceable = Mock()
                mock_import.return_value = fresh_langsmith
                
                result = _instrument_traceable_attributes(useOtel=True)
                
                # Should temporarily remove existing module, import fresh, and restore
                assert result == fresh_langsmith
        finally:
            # Restore original state  
            module.original_langsmith = original_state
    
    @patch('uipath_langchain.tracers._instrument_traceable.importlib.import_module')
    def test_import_error_handling(self, mock_import):
        """Test handling of import errors."""
        # Reset global state for this test
        import uipath_langchain.tracers._instrument_traceable as module
        original_state = module.original_langsmith
        module.original_langsmith = None
        
        try:
            mock_import.side_effect = ImportError("langsmith not found")
            
            # Current implementation propagates import errors
            with pytest.raises(ImportError, match="langsmith not found"):
                _instrument_traceable_attributes(useOtel=True)
        finally:
            # Restore original state
            module.original_langsmith = original_state


class TestIntegration:
    """Integration tests for the complete OTEL adapter flow."""
    
    @patch('uipath_langchain.tracers._instrument_traceable.traced')
    def test_end_to_end_workflow(self, mock_traced):
        """Test complete workflow from decorator to traced call."""
        # Setup mock
        mock_traced_decorator = Mock()
        mock_traced.return_value = mock_traced_decorator
        mock_traced_decorator.return_value = lambda f: f
        
        # Create and use decorator
        @otel_traceable_adapter(run_type="tool", name="search_tool")
        def search(query: str) -> str:
            return f"Results for {query}"
        
        # Verify traced was called with correct parameters
        mock_traced.assert_called_once()
        call_kwargs = mock_traced.call_args[1]
        
        assert call_kwargs["name"] == "search_tool"
        assert call_kwargs["run_type"] == "tool"
        assert call_kwargs["span_type"] == "tool_call"
    
    def test_parameter_validation(self):
        """Test parameter validation and edge cases."""
        # Test with None values
        result = _map_traceable_to_traced_args(
            name=None,
            run_type=None,
            tags=None,
            metadata=None
        )
        
        # Should handle None gracefully
        assert isinstance(result, dict)
        
        # Test with empty collections
        result = _map_traceable_to_traced_args(
            tags=[],
            metadata={}
        )
        
        # Should handle empty collections
        assert isinstance(result, dict)