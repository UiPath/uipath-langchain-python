import json
import os
import tempfile

import pytest

from uipath_langchain._cli.cli_run import langgraph_run_middleware


@pytest.fixture
def simple_agent() -> str:
    if os.path.isfile("mocks/simple_agent.py"):
        with open("mocks/simple_agent.py", "r") as file:
            data = file.read()
    else:
        with open("tests/cli/mocks/simple_agent.py", "r") as file:
            data = file.read()
    return data


@pytest.fixture
def uipath_json() -> str:
    if os.path.isfile("mocks/uipath.json"):
        with open("mocks/uipath.json", "r") as file:
            data = file.read()
    else:
        with open("tests/cli/mocks/uipath.json", "r") as file:
            data = file.read()
    return data


@pytest.fixture
def langgraph_json() -> str:
    if os.path.isfile("mocks/langgraph.json"):
        with open("mocks/langgraph.json", "r") as file:
            data = file.read()
    else:
        with open("tests/cli/mocks/langgraph.json", "r") as file:
            data = file.read()
    return data


class TestRun:
    def test_successful_execution(
        self,
        langgraph_json: str,
        uipath_json: str,
        simple_agent: str,
        mock_env_vars: dict[str, str],
    ):
        os.environ.clear()
        os.environ.update(mock_env_vars)
        input_file_name = "input.json"
        output_file_name = "output.json"
        agent_file_name = "main.py"
        input_json_content = {"topic": "UiPath"}
        with tempfile.TemporaryDirectory() as temp_dir:
            current_dir = os.getcwd()
            os.chdir(temp_dir)
            # Create input and output files
            input_file_path = os.path.join(temp_dir, input_file_name)
            output_file_path = os.path.join(temp_dir, output_file_name)

            with open(input_file_path, "w") as f:
                f.write(json.dumps(input_json_content))

            # Create test script
            script_file_path = os.path.join(temp_dir, agent_file_name)
            with open(script_file_path, "w") as f:
                f.write(simple_agent)

            # create uipath.json
            uipath_json_file_path = os.path.join(temp_dir, "uipath.json")
            with open(uipath_json_file_path, "w") as f:
                f.write(uipath_json)

            # Create langgraph.json
            langgraph_json_file_path = os.path.join(temp_dir, "langgraph.json")
            with open(langgraph_json_file_path, "w") as f:
                f.write(langgraph_json)

            result = langgraph_run_middleware(
                entrypoint="agent",
                input=None,
                resume=False,
                input_file=input_file_path,
                execution_output_file=output_file_path,
            )
            assert result.should_continue is False
            assert os.path.exists(output_file_path)
            with open(output_file_path, "r") as f:
                output = f.read()
                assert "This is mock report for" in output

            os.chdir(current_dir)
