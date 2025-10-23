import os
import tempfile

from uipath_langchain._cli.cli_init import (
    FileOperationStatus,
    generate_agent_md_file,
    generate_specific_agents_md_files,
)


class TestGenerateAgentMdFile:
    """Tests for the generate_agent_md_file function."""

    def test_generate_file_success(self):
        """Test successfully generating an agent MD file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_agent_md_file(
                temp_dir, "AGENTS.md", "uipath_langchain._resources", False
            )
            assert result is not None
            file_name, status = result
            assert file_name == "AGENTS.md"
            assert status == FileOperationStatus.CREATED

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

            result = generate_agent_md_file(
                temp_dir, "AGENTS.md", "uipath_langchain._resources", False
            )
            assert result is not None
            file_name, status = result
            assert file_name == "AGENTS.md"
            assert status == FileOperationStatus.UPDATED

            with open(target_path, "r") as f:
                content = f.read()

                assert content != original_content
                assert "Agent Code Patterns Reference" in content

    def test_generate_required_structure_file(self):
        """Test generating REQUIRED_STRUCTURE.md file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent_dir = os.path.join(temp_dir, ".agent")
            os.makedirs(agent_dir, exist_ok=True)
            result = generate_agent_md_file(
                agent_dir, "REQUIRED_STRUCTURE.md", "uipath_langchain._resources", False
            )
            assert result is not None
            file_name, status = result
            assert file_name == "REQUIRED_STRUCTURE.md"
            assert status == FileOperationStatus.CREATED

            target_path = os.path.join(agent_dir, "REQUIRED_STRUCTURE.md")
            assert os.path.exists(target_path)
            with open(target_path, "r") as f:
                content = f.read()
                assert "Required Agent Structure" in content

    def test_file_skipped_when_no_override(self):
        """Test that an existing file is skipped when no_agents_md_override is True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = os.path.join(temp_dir, "AGENTS.md")
            original_content = "Original content"
            with open(target_path, "w") as f:
                f.write(original_content)

            result = generate_agent_md_file(
                temp_dir, "AGENTS.md", "uipath_langchain._resources", True
            )
            assert result is not None
            file_name, status = result
            assert file_name == "AGENTS.md"
            assert status == FileOperationStatus.SKIPPED

            # Verify the file was not modified
            with open(target_path, "r") as f:
                content = f.read()
                assert content == original_content


class TestGenerateSpecificAgentsMdFiles:
    """Tests for the generate_specific_agents_md_files function."""

    def test_generate_all_files(self):
        """Test that all agent documentation files are generated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = list(generate_specific_agents_md_files(temp_dir, False))

            # Check that we got results for all files
            assert len(results) == 5
            file_names = [name for name, _ in results]
            assert "AGENTS.md" in file_names
            assert "REQUIRED_STRUCTURE.md" in file_names
            assert "CLAUDE.md" in file_names
            assert "CLI_REFERENCE.md" in file_names
            assert "SDK_REFERENCE.md" in file_names

            # Check all were created (not updated or skipped)
            for _, status in results:
                assert status == FileOperationStatus.CREATED

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

            results = list(generate_specific_agents_md_files(temp_dir, False))
            assert len(results) == 5
            assert os.path.exists(agent_dir)

    def test_files_overwritten(self):
        """Test that existing files are overwritten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agents_md_path = os.path.join(temp_dir, "AGENTS.md")
            original_content = "Custom documentation"
            with open(agents_md_path, "w") as f:
                f.write(original_content)

            results = list(generate_specific_agents_md_files(temp_dir, False))

            # Check that AGENTS.md was updated, others were created
            agents_result = [r for r in results if r[0] == "AGENTS.md"]
            assert len(agents_result) == 1
            _, status = agents_result[0]
            assert status == FileOperationStatus.UPDATED

            with open(agents_md_path, "r") as f:
                content = f.read()

                assert content != original_content
                assert "Agent Code Patterns Reference" in content

    def test_files_skipped_when_no_override(self):
        """Test that existing files are skipped when no_agents_md_override is True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some existing files
            agents_md_path = os.path.join(temp_dir, "AGENTS.md")
            claude_md_path = os.path.join(temp_dir, "CLAUDE.md")
            with open(agents_md_path, "w") as f:
                f.write("Existing AGENTS.md")
            with open(claude_md_path, "w") as f:
                f.write("Existing CLAUDE.md")

            results = list(generate_specific_agents_md_files(temp_dir, True))

            # Check that existing files were skipped
            skipped_files = [
                name
                for name, status in results
                if status == FileOperationStatus.SKIPPED
            ]
            assert "AGENTS.md" in skipped_files
            assert "CLAUDE.md" in skipped_files

            # Check that non-existing files were created
            created_files = [
                name
                for name, status in results
                if status == FileOperationStatus.CREATED
            ]
            assert "CLI_REFERENCE.md" in created_files
            assert "SDK_REFERENCE.md" in created_files
            assert "REQUIRED_STRUCTURE.md" in created_files

            # Verify the existing files were not modified
            with open(agents_md_path, "r") as f:
                assert f.read() == "Existing AGENTS.md"
            with open(claude_md_path, "r") as f:
                assert f.read() == "Existing CLAUDE.md"
