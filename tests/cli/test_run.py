import json
import os
import tempfile

import pytest
from uipath.runtime import UiPathRuntimeContext

from uipath_langchain.runtime import register_runtime_factory


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
    @pytest.mark.asyncio
    async def test_successful_execution(
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

        register_runtime_factory()

        with tempfile.TemporaryDirectory() as temp_dir:
            current_dir = os.getcwd()
            os.chdir(temp_dir)

            try:
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

                # Create runtime context
                context = UiPathRuntimeContext.with_defaults(
                    entrypoint="agent",
                    input=None,
                    input_file=input_file_path,
                    output_file=output_file_path,
                )

                # Get factory from registry (will auto-detect langgraph.json)
                from uipath.runtime import UiPathRuntimeFactoryRegistry

                factory = UiPathRuntimeFactoryRegistry.get(
                    search_path=temp_dir, context=context
                )

                # Create runtime
                runtime = await factory.new_runtime(
                    entrypoint="agent", runtime_id="test-runtime"
                )

                # Execute
                from uipath.runtime import UiPathExecuteOptions

                with context:
                    context.result = await runtime.execute(
                        input=input_json_content,
                        options=UiPathExecuteOptions(resume=False),
                    )

                # Verify results
                assert context.result is not None
                assert os.path.exists(output_file_path)
                with open(output_file_path, "r") as f:
                    output = f.read()
                    assert "This is mock report for" in output

                # Cleanup
                await runtime.dispose()
                await factory.dispose()

            finally:
                os.chdir(current_dir)
