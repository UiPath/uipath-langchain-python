import json
import os
import tempfile

import pytest

from uipath_langchain._cli.cli_init import (
    generate_agent_md_file,
    generate_specific_agents_md_files,
)
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


class TestGenerateAgentMdFile:
    """Tests for the generate_agent_md_file function."""

    def test_generate_file_success(self):
        """Test successfully generating an agent MD file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generate_agent_md_file(temp_dir, "AGENTS.md", "uipath_langchain._resources")
            target_path = os.path.join(temp_dir, "AGENTS.md")
            assert os.path.exists(target_path)
            with open(target_path, "r") as f:
                content = f.read()
                assert len(content) > 0
                assert "Agent Code Patterns Reference" in content

    def test_file_already_exists(self):
        """Test that an existing file is overwritten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = os.path.join(temp_dir, "AGENTS.md")
            original_content = "Original content"
            with open(target_path, "w") as f:
                f.write(original_content)

            generate_agent_md_file(temp_dir, "AGENTS.md", "uipath_langchain._resources")

            with open(target_path, "r") as f:
                content = f.read()

                assert content != original_content
                assert "Agent Code Patterns Reference" in content

    def test_generate_required_structure_file(self):
        """Test generating REQUIRED_STRUCTURE.md file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent_dir = os.path.join(temp_dir, ".agent")
            os.makedirs(agent_dir, exist_ok=True)
            generate_agent_md_file(
                agent_dir, "REQUIRED_STRUCTURE.md", "uipath_langchain._resources"
            )
            target_path = os.path.join(agent_dir, "REQUIRED_STRUCTURE.md")
            assert os.path.exists(target_path)
            with open(target_path, "r") as f:
                content = f.read()
                assert "Required Agent Structure" in content


class TestGenerateSpecificAgentsMdFiles:
    """Tests for the generate_specific_agents_md_files function."""

    def test_generate_all_files(self):
        """Test that all agent documentation files are generated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generate_specific_agents_md_files(temp_dir)

            agent_dir = os.path.join(temp_dir, ".agent")
            assert os.path.exists(agent_dir)
            assert os.path.isdir(agent_dir)

            agents_md_path = os.path.join(temp_dir, "AGENTS.md")
            assert os.path.exists(agents_md_path)

            required_structure_path = os.path.join(agent_dir, "REQUIRED_STRUCTURE.md")
            assert os.path.exists(required_structure_path)

            with open(agents_md_path, "r") as f:
                agents_content = f.read()
                assert "Agent Code Patterns Reference" in agents_content

            with open(required_structure_path, "r") as f:
                required_content = f.read()
                assert "Required Agent Structure" in required_content

    def test_agent_dir_already_exists(self):
        """Test that the existing .agent directory doesn't cause errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent_dir = os.path.join(temp_dir, ".agent")
            os.makedirs(agent_dir, exist_ok=True)

            generate_specific_agents_md_files(temp_dir)
            assert os.path.exists(agent_dir)

    def test_files_overwritten(self):
        """Test that existing files are overwritten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agents_md_path = os.path.join(temp_dir, "AGENTS.md")
            original_content = "Custom documentation"
            with open(agents_md_path, "w") as f:
                f.write(original_content)

            generate_specific_agents_md_files(temp_dir)

            with open(agents_md_path, "r") as f:
                content = f.read()

                assert content != original_content
                assert "Agent Code Patterns Reference" in content
