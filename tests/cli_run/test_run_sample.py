import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from uipath._cli._runtime._contracts import UiPathTraceContext

from uipath_langchain._cli._runtime._context import LangGraphRuntimeContext
from uipath_langchain._cli._runtime._runtime import LangGraphRuntime
from uipath_langchain._cli._utils._graph import LangGraphConfig

load_dotenv()


def _create_test_runtime_context(config: LangGraphConfig) -> LangGraphRuntimeContext:
    """Helper function to create and configure LangGraphRuntimeContext for tests."""
    context = LangGraphRuntimeContext.from_config(
        os.environ.get("UIPATH_CONFIG_PATH", "uipath.json")
    )

    context.entrypoint = (
        None  # Or a specific graph name if needed, None will pick the single one
    )
    context.input = '{ "graph_state": "GET Assets API does not enforce proper permissions Assets.View" }'
    context.resume = False
    context.langgraph_config = config
    context.logs_min_level = os.environ.get("LOG_LEVEL", "INFO")
    context.job_id = str(uuid.uuid4())
    context.trace_id = str(uuid.uuid4())
    # Convert string "True" or "False" to boolean for tracing_enabled
    tracing_enabled_str = os.environ.get("UIPATH_TRACING_ENABLED", "True")
    context.tracing_enabled = tracing_enabled_str.lower() == "true"
    context.trace_context = UiPathTraceContext(
        enabled=context.tracing_enabled,
        trace_id=str(
            uuid.uuid4()
        ),  # Consider passing trace_id if it needs to match context.trace_id
        parent_span_id=os.environ.get("UIPATH_PARENT_SPAN_ID"),
        root_span_id=os.environ.get("UIPATH_ROOT_SPAN_ID"),
        job_id=os.environ.get(
            "UIPATH_JOB_KEY"
        ),  # Consider passing job_id if it needs to match context.job_id
        org_id=os.environ.get("UIPATH_ORGANIZATION_ID"),
        tenant_id=os.environ.get("UIPATH_TENANT_ID"),
        process_key=os.environ.get("UIPATH_PROCESS_UUID"),
        folder_key=os.environ.get("UIPATH_FOLDER_KEY"),
    )
    # Convert string "True" or "False" to boolean
    langsmith_tracing_enabled_str = os.environ.get("LANGSMITH_TRACING", "False")
    context.langsmith_tracing_enabled = langsmith_tracing_enabled_str.lower() == "true"
    return context


@pytest.mark.asyncio
async def test_langgraph_runtime():
    test_folder_path = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(test_folder_path, "samples", "1-simple-graph")

    sys.path.append(sample_path)
    os.chdir(sample_path)

    config = LangGraphConfig()
    if not config.exists:
        raise AssertionError("langgraph.json not found in sample path")

    context = _create_test_runtime_context(config)

    # Mocking UiPath SDK for action creation
    with patch("uipath_langchain._cli._runtime._output.UiPath") as MockUiPathClass:
        mock_uipath_sdk_instance = MagicMock()
        MockUiPathClass.return_value = mock_uipath_sdk_instance
        mock_actions_client = MagicMock()
        mock_uipath_sdk_instance.actions = mock_actions_client

        mock_created_action = MagicMock()
        mock_created_action.key = "mock_action_key_from_test"
        mock_actions_client.create.return_value = mock_created_action

        result = None
        async with LangGraphRuntime.from_context(context) as runtime:
            result = await runtime.execute()
            print("Result:", result)

        context.resume = True
        context.input = '{ "answer":  "John Doe"}'  # Simulate some resume data
        async with LangGraphRuntime.from_context(context) as runtime:
            result = await runtime.execute()
            print("Result:", result)

        context.resume = True
        context.input = (
            '{ "ActionData": "Test-ActionData" }'  # Simulate some resume data
        )
        async with LangGraphRuntime.from_context(context) as runtime:
            result = await runtime.execute()
            print("Result:", result)

        assert result is not None, "Result should not be None after execution"
        assert (
            result.output["graph_state"] == "Hello, I am John Doe!Test-ActionData end"
        )
