"""Pytest configuration for agent_graph_builder tests."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel


@pytest.fixture(autouse=True)
def mock_uipath_llm_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock UiPath LangChain classes to avoid requiring credentials."""

    class MockChatModel(BaseChatModel):
        """Mock chat model for testing."""

        model_name: str = "mock-model"

        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            """Mock generation method."""
            return MagicMock()

        @property
        def _llm_type(self) -> str:
            """Return mock LLM type."""
            return "mock"

        @property
        def _identifying_params(self) -> dict[str, Any]:
            """Return identifying parameters."""
            return {"model_name": self.model_name}

    def create_mock_llm(*args: Any, **kwargs: Any) -> BaseChatModel:
        """Create a mock LLM that properly inherits from BaseChatModel."""
        return MockChatModel()

    monkeypatch.setattr(
        "uipath_langchain.chat.openai.UiPathChatOpenAI",
        create_mock_llm,
    )
    monkeypatch.setattr(
        "uipath_langchain.chat.bedrock.UiPathChatBedrockConverse",
        create_mock_llm,
    )
    monkeypatch.setattr(
        "uipath_langchain.chat.vertex.UiPathChatVertex",
        create_mock_llm,
    )
