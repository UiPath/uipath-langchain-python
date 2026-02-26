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
        """Test successfully generating CLAUDE.md file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_agent_md_file(
                temp_dir, "CLAUDE.md", "uipath_langchain._resources", False
            )
            assert result is not None
            file_name, status = result
            assert file_name == "CLAUDE.md"
            assert status == FileOperationStatus.CREATED

            target_path = os.path.join(temp_dir, "CLAUDE.md")
            assert os.path.exists(target_path)
            with open(target_path, "r") as f:
                content = f.read()
                assert len(content) > 0
                assert "LangGraph" in content

    def test_file_already_exists(self):
        """Test that an existing file is overwritten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = os.path.join(temp_dir, "CLAUDE.md")
            original_content = "Original content"
            with open(target_path, "w") as f:
                f.write(original_content)

            result = generate_agent_md_file(
                temp_dir, "CLAUDE.md", "uipath_langchain._resources", False
            )
            assert result is not None
            file_name, status = result
            assert file_name == "CLAUDE.md"
            assert status == FileOperationStatus.UPDATED

            with open(target_path, "r") as f:
                content = f.read()

                assert content != original_content
                assert "LangGraph" in content

    def test_file_skipped_when_no_override(self):
        """Test that an existing file is skipped when no_agents_md_override is True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = os.path.join(temp_dir, "CLAUDE.md")
            original_content = "Original content"
            with open(target_path, "w") as f:
                f.write(original_content)

            result = generate_agent_md_file(
                temp_dir, "CLAUDE.md", "uipath_langchain._resources", True
            )
            assert result is not None
            file_name, status = result
            assert file_name == "CLAUDE.md"
            assert status == FileOperationStatus.SKIPPED

            # Verify the file was not modified
            with open(target_path, "r") as f:
                content = f.read()
                assert content == original_content


class TestGenerateSpecificAgentsMdFiles:
    """Tests for the generate_specific_agents_md_files function."""

    def test_generate_only_claude_md(self):
        """Test that only CLAUDE.md is generated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = list(generate_specific_agents_md_files(temp_dir, False))

            # Check that we got result only for CLAUDE.md
            assert len(results) == 1
            file_names = [name for name, _ in results]
            assert "CLAUDE.md" in file_names

            # Should NOT create AGENTS.md or .agent directory
            assert not os.path.exists(os.path.join(temp_dir, "AGENTS.md"))
            assert not os.path.exists(os.path.join(temp_dir, ".agent"))

            # Check it was created (not updated or skipped)
            for _, status in results:
                assert status == FileOperationStatus.CREATED

            claude_md_path = os.path.join(temp_dir, "CLAUDE.md")
            assert os.path.exists(claude_md_path)

            with open(claude_md_path, "r") as f:
                claude_content = f.read()
                assert "LangGraph" in claude_content

    def test_files_overwritten(self):
        """Test that existing CLAUDE.md is overwritten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            claude_md_path = os.path.join(temp_dir, "CLAUDE.md")
            original_content = "Custom documentation"
            with open(claude_md_path, "w") as f:
                f.write(original_content)

            results = list(generate_specific_agents_md_files(temp_dir, False))

            # Check that CLAUDE.md was updated
            claude_result = [r for r in results if r[0] == "CLAUDE.md"]
            assert len(claude_result) == 1
            _, status = claude_result[0]
            assert status == FileOperationStatus.UPDATED

            with open(claude_md_path, "r") as f:
                content = f.read()

                assert content != original_content
                assert "LangGraph" in content

    def test_files_skipped_when_no_override(self):
        """Test that existing CLAUDE.md is skipped when no_agents_md_override is True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create existing CLAUDE.md
            claude_md_path = os.path.join(temp_dir, "CLAUDE.md")
            with open(claude_md_path, "w") as f:
                f.write("Existing CLAUDE.md")

            results = list(generate_specific_agents_md_files(temp_dir, True))

            # Check that existing file was skipped
            skipped_files = [
                name
                for name, status in results
                if status == FileOperationStatus.SKIPPED
            ]
            assert "CLAUDE.md" in skipped_files

            # Verify the existing file was not modified
            with open(claude_md_path, "r") as f:
                assert f.read() == "Existing CLAUDE.md"
