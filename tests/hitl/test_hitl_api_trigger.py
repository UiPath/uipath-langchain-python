import json
import os
import shutil
import sqlite3

from click.testing import CliRunner
from pytest_httpx import HTTPXMock

from tests.hitl.conftest import get_file_path
from uipath_langchain._cli.cli_run import langgraph_run_middleware


class TestHitlApiTrigger:
    """Test class for HITL API trigger functionality."""

    def test_agent(
        self,
        runner: CliRunner,
        temp_dir: str,
        httpx_mock: HTTPXMock,
        setup_test_env: None,
    ) -> None:
        script_name = "api_trigger_hitl.py"
        script_file_path = get_file_path(script_name)

        config_file_name = "uipath.json"
        config_file_path = get_file_path(config_file_name)

        langgraph_config_file_name = "langgraph.json"
        langgraph_config_file_path = get_file_path(langgraph_config_file_name)

        with runner.isolated_filesystem(temp_dir=temp_dir):
            # Copy the API trigger test file to our temp directory

            shutil.copy(script_file_path, "hitl.py")
            shutil.copy(config_file_path, config_file_name)
            shutil.copy(langgraph_config_file_path, langgraph_config_file_name)

            result = langgraph_run_middleware("agent", "{}", False)

            assert result.error_message is None

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

                # Check the inserted trigger
                cursor.execute("SELECT * FROM __uipath_resume_triggers")
                triggers = cursor.fetchall()
                assert len(triggers) == 1
                _, type, key, folder_path, folder_key, payload, _ = triggers[0]
                assert type == "Api"
                assert folder_path == folder_key is None
                assert payload == "interrupt message"
            finally:
                if conn:
                    conn.close()

            # test resume scenario with mocked api response
            base_url = os.getenv("UIPATH_URL")
            httpx_mock.add_response(
                url=f"{base_url}/orchestrator_/api/JobTriggers/GetPayload/{key}",
                status_code=200,
                text=json.dumps({"payload": "human response"}),
            )
            result = langgraph_run_middleware("agent", "{}", True)
            assert result.error_message is None
            assert result.should_continue is False

            # check output.json for api trigger response
            with open("__uipath/output.json", "r") as f:
                output = f.read()
            json_output = json.loads(output)
            assert json_output["output"] == {"message": "human response"}
