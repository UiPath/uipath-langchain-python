import json
import os
import os.path
import shutil
import sqlite3
import uuid

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


class TestHitlJobTrigger:
    """Test class for Job trigger functionality."""

    @pytest.mark.asyncio
    async def test_agent_job_trigger(
        self,
        runner: CliRunner,
        temp_dir: str,
        httpx_mock: HTTPXMock,
        setup_test_env: None,
    ) -> None:
        register_runtime_factory()
        script_name = "job_trigger_hitl.py"
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
                shutil.copy(langgraph_config_file_path, "langgraph.json")

                # mock app creation
                base_url = os.getenv("UIPATH_URL")
                job_key = uuid.uuid4()

                # Mock UiPath API response for job creation
                httpx_mock.add_response(
                    url=f"{base_url}/orchestrator_/odata/Jobs/UiPath.Server.Configuration.OData.StartJobs",
                    json={"value": [{"key": f"{job_key}", "Id": "123"}]},
                )

                # First execution: creates job trigger and stores it in database
                context = UiPathRuntimeContext.with_defaults(
                    entrypoint="agent",
                    input="{}",
                    output_file="__uipath/output.json",
                )

                factory = UiPathRuntimeFactoryRegistry.get(
                    search_path=os.getcwd(), context=context
                )

                runtime = await factory.new_runtime(
                    entrypoint="agent", runtime_id="test-job-runtime"
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

                    # Check the first job trigger data
                    cursor.execute(
                        "SELECT runtime_id, interrupt_id, data FROM __uipath_resume_triggers"
                    )
                    triggers = cursor.fetchall()
                    assert len(triggers) == 1
                    runtime_id, interrupt_id, data = triggers[0]

                    trigger_data = json.loads(data)
                    assert trigger_data["trigger_type"] == "Job"
                    assert trigger_data["trigger_name"] == "Job"
                    assert trigger_data["folder_path"] == "process-folder-path"
                    assert trigger_data["folder_key"] is None
                    assert "input_arg_1" in data
                    assert "value_1" in data
                finally:
                    if conn:
                        conn.close()

                # Cleanup first runtime
                await runtime.dispose()
                await factory.dispose()

                # Mock response for first resume: job output arguments
                output_args_dict = {"output_arg_1": "response from invoke process"}
                httpx_mock.add_response(
                    url=f"{base_url}/orchestrator_/odata/Jobs/UiPath.Server.Configuration.OData.GetByKey(identifier={trigger_data['item_key']})",
                    json={
                        "key": f"{job_key}",
                        "id": 123,
                        "state": "successful",
                        "output_arguments": json.dumps(output_args_dict),
                    },
                )

                # Second execution: resume from first trigger
                resume_context_1 = UiPathRuntimeContext.with_defaults(
                    entrypoint="agent",
                    input="{}",
                    output_file="__uipath/output.json",
                    resume=True,
                )

                resume_factory_1 = UiPathRuntimeFactoryRegistry.get(
                    search_path=os.getcwd(), context=resume_context_1
                )

                resume_runtime_1 = await resume_factory_1.new_runtime(
                    entrypoint="agent", runtime_id="test-job-runtime"
                )

                with resume_context_1:
                    resume_context_1.result = await resume_runtime_1.execute(
                        input=None, options=UiPathExecuteOptions(resume=True)
                    )

                assert resume_context_1.result is not None

                # Verify second trigger information
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

                    # Check the second job trigger data (from wait job)
                    cursor.execute("""SELECT runtime_id, interrupt_id, data FROM __uipath_resume_triggers
                                    ORDER BY timestamp DESC
                                    """)
                    triggers = cursor.fetchall()
                    assert len(triggers) == 1
                    runtime_id, interrupt_id, data = triggers[0]

                    trigger_data = json.loads(data)
                    assert trigger_data["trigger_type"] == "Job"
                    assert trigger_data["trigger_name"] == "Job"
                    assert trigger_data["folder_path"] is None
                    assert trigger_data["folder_key"] is None
                    assert "123" in data
                    assert (
                        trigger_data["item_key"]
                        == "487d9dc7-30fe-4926-b5f0-35a956914042"
                    )
                finally:
                    if conn:
                        conn.close()

                # Cleanup second runtime
                await resume_runtime_1.dispose()
                await resume_factory_1.dispose()

                # Mock response for second resume: wait job output arguments
                output_args_dict = {"output_arg_2": "response from wait job"}

                httpx_mock.add_response(
                    url=f"{base_url}/orchestrator_/odata/Jobs/UiPath.Server.Configuration.OData.GetByKey(identifier={trigger_data['item_key']})",
                    json={
                        "key": f"{job_key}",
                        "id": 123,
                        "state": "successful",
                        "output_arguments": json.dumps(output_args_dict),
                    },
                )

                # Third execution: resume from second trigger and complete
                resume_context_2 = UiPathRuntimeContext.with_defaults(
                    entrypoint="agent",
                    input="{}",
                    output_file="__uipath/output.json",
                    resume=True,
                )

                resume_factory_2 = UiPathRuntimeFactoryRegistry.get(
                    search_path=os.getcwd(), context=resume_context_2
                )

                resume_runtime_2 = await resume_factory_2.new_runtime(
                    entrypoint="agent", runtime_id="test-job-runtime"
                )

                with resume_context_2:
                    resume_context_2.result = await resume_runtime_2.execute(
                        input=None, options=UiPathExecuteOptions(resume=True)
                    )

                assert resume_context_2.result is not None

                # Verify final output contains the last job response
                with open("__uipath/output.json", "r") as f:
                    output = f.read()
                json_output = json.loads(output)
                assert json_output == {"message": "response from wait job"}

                # Verify execution log contains both job responses
                with open("__uipath/execution.log", "r") as f:
                    output = f.read()

                assert "Process output" in output
                assert "response from invoke process" in output

                # Cleanup
                await resume_runtime_2.dispose()
                await resume_factory_2.dispose()

            finally:
                os.chdir(current_dir)
