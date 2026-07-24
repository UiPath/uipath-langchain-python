"""Coded agents exercised by the model-onboarding testcase.

Each agent is the coded (LangGraph) equivalent of a low-code UiPath agent,
re-authored here so the testcase can run it against a configurable model +
API flavor (sourced from ``model_spec`` in ``input.json``).

Add a new coded agent by creating ``agents/<name>/agent.py`` exposing an async
``run(model, prompt, files)`` coroutine and a ``NAME`` constant, then register
it in ``AGENT_REGISTRY`` below. Every registered agent becomes a per-path cell
in the onboarding grid.
"""

from typing import Awaitable, Callable

from langchain_core.language_models import BaseChatModel

from uipath_langchain.agent.multimodal.types import FileInfo

from .file_processing.agent import NAME as FILE_PROCESSING_NAME
from .file_processing.agent import run as run_file_processing

# An agent entry: given a built model, the prompt, and the selected files,
# returns a result string ("✓" or "✗ ...").
AgentRunner = Callable[[BaseChatModel, str, list[FileInfo]], Awaitable[str]]

AGENT_REGISTRY: dict[str, AgentRunner] = {
    FILE_PROCESSING_NAME: run_file_processing,
}

__all__ = ["AGENT_REGISTRY", "AgentRunner"]
