import json
import os
import shutil
import sqlite3

import pytest
from click.testing import CliRunner
from pytest_httpx import HTTPXMock
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeFactoryRegistry,
)

from tests.hitl.conftest import get_file_path
from uipath_langchain.runtime import register_runtime_factory


class TestHitlApiTrigger:
    """Test class for HITL API trigger functionality."""

    @pytest.mark.asyncio
    async def test_agent(
        self,
        runner: CliRunner,
        temp_dir: str,
        httpx_mock: HTTPXMock,
        setup_test_env: None,
    ) -> None:
        register_runtime_factory()

        script_name = "api_trigger_hitl.py"
        script_file_path = get_file_path(script_name)

        config_file_name = "uipath.json"
        config_file_path = get_file_path(config_file_name)

        langgraph_config_file_name = "langgraph.json"
        langgraph_config_file_path = get_file_path(langgraph_config_file_name)

        with runner.isolated_filesystem(temp_dir=temp_dir):
            current_dir = os.getcwd()

            try:
                # Copy the API trigger test file to our temp directory
                shutil.copy(script_file_path, "hitl.py")
                shutil.copy(config_file_path, config_file_name)
                shutil.copy(langgraph_config_file_path, langgraph_config_file_name)

                # First execution: creates interrupt and stores trigger in database
                context = UiPathRuntimeContext.with_defaults(
                    entrypoint="agent",
                    input="{}",
                    output_file="__uipath/output.json",
                )

                factory = UiPathRuntimeFactoryRegistry.get(
                    search_path=os.getcwd(), context=context
                )

                runtime = await factory.new_runtime(
                    entrypoint="agent", runtime_id="test-hitl-runtime"
                )

                with context:
                    context.result = await runtime.execute(
                        input=None, options=UiPathExecuteOptions(resume=False)
                    )

                assert context.result is not None

                # Verify that __uipath directory and state.db were created
                assert os.path.exists("__uipath")
                assert os.path.exists("__uipath/state.db")

                # Verify the state database contains trigger information
                conn = None
                try:
                    conn = sqlite3.connect("__uipath/state.db")
                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name='__uipath_resume_triggers'
                    """)
                    tables = cursor.fetchall()
                    assert len(tables) == 1

                    # Check the inserted trigger data from first execution
                    cursor.execute(
                        "SELECT runtime_id, interrupt_id, data FROM __uipath_resume_triggers"
                    )
                    triggers = cursor.fetchall()
                    assert len(triggers) == 1
                    runtime_id, interrupt_id, data = triggers[0]

                    # Parse the JSON data
                    trigger_data = json.loads(data)
                    assert trigger_data["trigger_type"] == "Api"
                    assert trigger_data["trigger_name"] == "Api"
                    assert trigger_data["folder_path"] is None
                    assert trigger_data["folder_key"] is None
                    assert trigger_data["payload"] == "interrupt message"
                finally:
                    if conn:
                        conn.close()

                # Cleanup first runtime
                await runtime.dispose()
                await factory.dispose()

                # Mock API response for resume scenario
                base_url = os.getenv("UIPATH_URL")
                httpx_mock.add_response(
                    url=f"{base_url}/orchestrator_/api/JobTriggers/GetPayload/{trigger_data['api_resume']['inbox_id']}",
                    status_code=200,
                    text=json.dumps({"payload": "human response"}),
                )

                # Second execution: resume from stored trigger and fetch human response
                resume_context = UiPathRuntimeContext.with_defaults(
                    entrypoint="agent",
                    input="{}",
                    output_file="__uipath/output.json",
                    resume=True,
                )

                resume_factory = UiPathRuntimeFactoryRegistry.get(
                    search_path=os.getcwd(), context=resume_context
                )

                resume_runtime = await resume_factory.new_runtime(
                    entrypoint="agent", runtime_id="test-hitl-runtime"
                )

                with resume_context:
                    resume_context.result = await resume_runtime.execute(
                        input=None, options=UiPathExecuteOptions(resume=True)
                    )

                assert resume_context.result is not None

                # Verify the final output contains the resumed data
                with open("__uipath/output.json", "r") as f:
                    output = f.read()
                json_output = json.loads(output)
                assert json_output == {"message": "human response"}

                # Cleanup
                await resume_runtime.dispose()
                await resume_factory.dispose()

            finally:
                os.chdir(current_dir)
