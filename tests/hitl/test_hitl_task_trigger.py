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


def get_org_scoped_url(base_url: str):
    return base_url.rsplit("/", 1)[0]


class TestHitlActionTrigger:
    """Test class for Action trigger functionality."""

    @pytest.mark.asyncio
    async def test_agent_action_trigger(
        self,
        runner: CliRunner,
        temp_dir: str,
        httpx_mock: HTTPXMock,
        setup_test_env: None,
    ) -> None:
        register_runtime_factory()
        script_name = "action_trigger_hitl.py"
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
                base_url = os.getenv(
                    "UIPATH_URL", "https://cloud.uipath.com/organization/tenant"
                )
                action_key = uuid.uuid4()

                # Mock UiPath API responses for action app creation
                httpx_mock.add_response(
                    url=f"{base_url}/orchestrator_/tasks/AppTasks/CreateAppTask",
                    json={
                        "id": 1,
                        "title": "Action Required: Report Review",
                        "key": f"{action_key}",
                    },
                )

                httpx_mock.add_response(
                    url=f"{get_org_scoped_url(base_url)}/apps_/default/api/v1/default/deployed-action-apps-schemas?search=HITL APP&filterByDeploymentTitle=true",
                    text=json.dumps(
                        {
                            "deployed": [
                                {
                                    "deploymentFolder": {
                                        "fullyQualifiedName": "app-folder-path"
                                    },
                                    "systemName": "HITL APP",
                                    "actionSchema": {
                                        "key": "test-key",
                                        "inputs": [],
                                        "outputs": [],
                                        "inOuts": [],
                                        "outcomes": [],
                                    },
                                }
                            ]
                        }
                    ),
                )

                # First execution: creates action trigger and stores it in database
                context = UiPathRuntimeContext.with_defaults(
                    entrypoint="agent",
                    input="{}",
                    output_file="__uipath/output.json",
                )

                factory = UiPathRuntimeFactoryRegistry.get(
                    search_path=os.getcwd(), context=context
                )

                runtime = await factory.new_runtime(
                    entrypoint="agent", runtime_id="test-action-runtime"
                )

                with context:
                    context.result = await runtime.execute(
                        input={}, options=UiPathExecuteOptions(resume=False)
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

                    # Check the first action trigger data
                    cursor.execute(
                        "SELECT runtime_id, interrupt_id, data FROM __uipath_resume_triggers"
                    )
                    triggers = cursor.fetchall()
                    assert len(triggers) == 1
                    runtime_id, interrupt_id, data = triggers[0]

                    trigger_data = json.loads(data)
                    assert trigger_data["trigger_type"] == "Task"
                    assert trigger_data["folder_path"] == "app-folder-path"
                    assert trigger_data["folder_key"] is None
                    assert "agent question" in data
                    assert "Action Required" in data
                finally:
                    if conn:
                        conn.close()

                # Cleanup first runtime
                await runtime.dispose()
                await factory.dispose()

                # Mock response for first resume: human response from create action
                httpx_mock.add_response(
                    url=f"{base_url}/orchestrator_/tasks/GenericTasks/GetTaskDataByKey?taskKey={trigger_data['item_key']}",
                    json={
                        "id": 1,
                        "title": "Action Required: Report Review",
                        "data": {"Answer": "human response from create action"},
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
                    entrypoint="agent", runtime_id="test-action-runtime"
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

                    # Check the second trigger data (from wait action)
                    cursor.execute("""SELECT runtime_id, interrupt_id, data FROM __uipath_resume_triggers
                                    ORDER BY timestamp DESC
                                    """)
                    triggers = cursor.fetchall()
                    assert len(triggers) == 1
                    runtime_id, interrupt_id, data = triggers[0]

                    trigger_data = json.loads(data)
                    assert trigger_data["trigger_type"] == "Task"
                    assert trigger_data["trigger_name"] == "Task"
                    assert trigger_data["folder_path"] is None
                    assert trigger_data["folder_key"] is None
                    assert "agent question from wait action" in data
                    assert (
                        trigger_data["item_key"]
                        == "1662478a-65b4-4a09-8e22-d707e5bd64f3"
                    )
                finally:
                    if conn:
                        conn.close()

                # Cleanup second runtime
                await resume_runtime_1.dispose()
                await resume_factory_1.dispose()

                # Mock response for second resume: human response from wait action
                httpx_mock.add_response(
                    url=f"{base_url}/orchestrator_/tasks/GenericTasks/GetTaskDataByKey?taskKey={trigger_data['item_key']}",
                    json={
                        "id": 1,
                        "title": "Action Required: Report Review",
                        "data": {"Answer": "human response from wait action"},
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
                    entrypoint="agent", runtime_id="test-action-runtime"
                )

                with resume_context_2:
                    resume_context_2.result = await resume_runtime_2.execute(
                        input=None, options=UiPathExecuteOptions(resume=True)
                    )

                assert resume_context_2.result is not None

                # Verify final output contains the last human response
                with open("__uipath/output.json", "r") as f:
                    output = f.read()
                json_output = json.loads(output)
                assert json_output == {"message": "human response from wait action"}

                # Verify execution log contains both human responses
                with open("__uipath/execution.log", "r") as f:
                    output = f.read()

                assert "Response from HITL action:" in output
                assert "human response from create action" in output

                # Cleanup
                await resume_runtime_2.dispose()
                await resume_factory_2.dispose()

            finally:
                os.chdir(current_dir)
