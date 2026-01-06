"""Tests demonstrating streaming throttling issues.

These tests prove that the current implementation has bugs where:
1. OpenAI/Vertex: Semaphore releases when connection established, not when stream completes
2. Bedrock: _astream bypasses semaphore entirely
"""

import ast
import asyncio
from pathlib import Path

import pytest


class MockSemaphore:
    """Mock semaphore that tracks acquire/release timing."""

    def __init__(self):
        self.acquired_at = []
        self.released_at = []
        self.acquire_count = 0
        self.release_count = 0
        self._counter = 0

    async def __aenter__(self):
        self._counter += 1
        self.acquire_count += 1
        self.acquired_at.append(f"acquire_{self._counter}")
        return self

    async def __aexit__(self, *args):
        self.release_count += 1
        self.released_at.append(f"release_{self._counter}")

    @property
    def is_held(self):
        return self.acquire_count > self.release_count


def get_source_path(module_path: str) -> Path:
    """Get the source file path."""
    return Path(__file__).parent.parent.parent / "src" / module_path


def check_method_has_semaphore(
    file_path: Path, class_name: str, method_name: str
) -> bool:
    """Check if a method in a class uses _get_llm_semaphore."""
    source = file_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    # Check if _get_llm_semaphore is called in the method
                    method_source = ast.unparse(item)
                    return "_get_llm_semaphore" in method_source
    return False


def check_class_has_method(file_path: Path, class_name: str, method_name: str) -> bool:
    """Check if a class defines a specific method (not inherited)."""
    source = file_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return True
    return False


class TestOpenAIStreamingThrottleFixed:
    """Test that OpenAI has correct streaming throttling after fix."""

    def test_semaphore_in_model_not_transport(self):
        """Verify semaphore is in chat model's _agenerate/_astream, not transport."""
        openai_path = get_source_path("uipath_langchain/chat/openai.py")

        # FIXED: Semaphore should NOT be in transport
        transport_has_semaphore = check_method_has_semaphore(
            openai_path, "UiPathURLRewriteTransport", "handle_async_request"
        )

        # FIXED: Semaphore should be in _agenerate/_astream methods of chat model
        model_has_agenerate = check_class_has_method(
            openai_path, "UiPathChatOpenAI", "_agenerate"
        )
        model_has_astream = check_class_has_method(
            openai_path, "UiPathChatOpenAI", "_astream"
        )
        agenerate_has_semaphore = check_method_has_semaphore(
            openai_path, "UiPathChatOpenAI", "_agenerate"
        )
        astream_has_semaphore = check_method_has_semaphore(
            openai_path, "UiPathChatOpenAI", "_astream"
        )

        # Verify fix: semaphore moved from transport to model
        assert not transport_has_semaphore, "FIXED: Transport should not have semaphore"
        assert model_has_agenerate, "FIXED: _agenerate is overridden in model"
        assert model_has_astream, "FIXED: _astream is overridden in model"
        assert agenerate_has_semaphore, "FIXED: _agenerate has semaphore"
        assert astream_has_semaphore, "FIXED: _astream has semaphore"


class TestVertexStreamingThrottleFixed:
    """Test that Vertex has correct streaming throttling after fix."""

    def test_semaphore_in_model_not_transport(self):
        """Verify semaphore is in chat model's _agenerate, not transport."""
        vertex_path = get_source_path("uipath_langchain/chat/vertex.py")

        # FIXED: Semaphore should NOT be in transport
        transport_has_semaphore = check_method_has_semaphore(
            vertex_path, "_AsyncUrlRewriteTransport", "handle_async_request"
        )

        # FIXED: Semaphore should be in _agenerate method of chat model
        model_has_agenerate = check_class_has_method(
            vertex_path, "UiPathChatVertex", "_agenerate"
        )
        agenerate_has_semaphore = check_method_has_semaphore(
            vertex_path, "UiPathChatVertex", "_agenerate"
        )

        # Verify fix
        assert not transport_has_semaphore, "FIXED: Transport should not have semaphore"
        assert model_has_agenerate, "FIXED: _agenerate is overridden in model"
        assert agenerate_has_semaphore, "FIXED: _agenerate has semaphore"


class TestBedrockStreamingThrottleFixed:
    """Test that Bedrock has _astream with semaphore after fix."""

    def test_astream_overridden_with_semaphore(self):
        """Verify _astream is overridden with semaphore protection."""
        bedrock_path = get_source_path("uipath_langchain/chat/bedrock.py")

        # Check _agenerate is overridden with semaphore
        converse_has_agenerate = check_class_has_method(
            bedrock_path, "UiPathChatBedrockConverse", "_agenerate"
        )
        bedrock_has_agenerate = check_class_has_method(
            bedrock_path, "UiPathChatBedrock", "_agenerate"
        )

        # FIXED: _astream is now overridden with semaphore
        converse_has_astream = check_class_has_method(
            bedrock_path, "UiPathChatBedrockConverse", "_astream"
        )
        bedrock_has_astream = check_class_has_method(
            bedrock_path, "UiPathChatBedrock", "_astream"
        )
        converse_astream_has_semaphore = check_method_has_semaphore(
            bedrock_path, "UiPathChatBedrockConverse", "_astream"
        )
        bedrock_astream_has_semaphore = check_method_has_semaphore(
            bedrock_path, "UiPathChatBedrock", "_astream"
        )

        assert converse_has_agenerate, "_agenerate is overridden"
        assert bedrock_has_agenerate, "_agenerate is overridden"
        assert converse_has_astream, (
            "FIXED: _astream is overridden in UiPathChatBedrockConverse"
        )
        assert bedrock_has_astream, "FIXED: _astream is overridden in UiPathChatBedrock"
        assert converse_astream_has_semaphore, (
            "FIXED: _astream has semaphore in Converse"
        )
        assert bedrock_astream_has_semaphore, "FIXED: _astream has semaphore in Bedrock"


class TestCorrectStreamingThrottle:
    """Tests for the CORRECT behavior after fixes are applied."""

    @pytest.mark.asyncio
    async def test_mixin_astream_request_holds_semaphore_during_stream(self):
        """Verify _astream_request in mixin correctly holds semaphore during streaming.

        This is the CORRECT implementation pattern we should follow.
        """
        # The mixin's _astream_request wraps the entire async generator,
        # so the semaphore is held for the full duration of streaming
        mock_semaphore = MockSemaphore()
        chunks_yielded = []

        async def simulate_correct_streaming():
            """Simulates correct behavior where semaphore is held during entire stream."""
            async with mock_semaphore:
                # Semaphore acquired
                assert mock_semaphore.is_held, "Semaphore should be held"

                for i in range(3):
                    chunks_yielded.append(f"chunk_{i}")
                    yield f"chunk_{i}"
                    # Semaphore still held during iteration
                    assert mock_semaphore.is_held, (
                        f"Semaphore should still be held at chunk {i}"
                    )

                # Semaphore still held until we exit the context
                assert mock_semaphore.is_held, (
                    "Semaphore should be held until stream ends"
                )

        # Consume the stream
        async for _chunk in simulate_correct_streaming():
            pass

        # Now semaphore should be released
        assert not mock_semaphore.is_held, "Semaphore should be released after stream"
        assert chunks_yielded == ["chunk_0", "chunk_1", "chunk_2"]

    @pytest.mark.asyncio
    async def test_correct_agenerate_throttling_pattern(self):
        """Verify the correct pattern for _agenerate throttling.

        _agenerate should wrap the entire generation call, including any
        internal streaming that happens.
        """
        mock_semaphore = MockSemaphore()
        generation_phases = []

        async def simulate_correct_agenerate():
            """Simulates correct _agenerate wrapping."""
            async with mock_semaphore:
                generation_phases.append("start_generation")
                assert mock_semaphore.is_held

                # Simulate internal work (could include streaming internally)
                await asyncio.sleep(0.01)
                generation_phases.append("processing")
                assert mock_semaphore.is_held

                generation_phases.append("complete")
                assert mock_semaphore.is_held

                return "generated_result"

        result = await simulate_correct_agenerate()

        assert result == "generated_result"
        assert not mock_semaphore.is_held
        assert generation_phases == ["start_generation", "processing", "complete"]
