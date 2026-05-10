import os
import tempfile

from uipath_langchain._cli.cli_init import (
    FileOperationStatus,
    generate_agent_md_file,
    generate_specific_agents_md_files,
)


class TestGenerateAgentMdFile:
    """Tests for the generate_agent_md_file helper."""

    def test_generate_file_success(self):
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
            with open(target_path) as f:
                content = f.read()
            assert "uip skills install" in content

    def test_existing_file_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = os.path.join(temp_dir, "AGENTS.md")
            with open(target_path, "w") as f:
                f.write("original")

            result = generate_agent_md_file(
                temp_dir, "AGENTS.md", "uipath_langchain._resources", False
            )
            assert result == ("AGENTS.md", FileOperationStatus.UPDATED)

            with open(target_path) as f:
                assert f.read() != "original"

    def test_existing_file_skipped_when_no_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = os.path.join(temp_dir, "AGENTS.md")
            with open(target_path, "w") as f:
                f.write("user content")

            result = generate_agent_md_file(
                temp_dir, "AGENTS.md", "uipath_langchain._resources", True
            )
            assert result == ("AGENTS.md", FileOperationStatus.SKIPPED)

            with open(target_path) as f:
                assert f.read() == "user content"


class TestGenerateSpecificAgentsMdFiles:
    """Tests for the generate_specific_agents_md_files entry point."""

    def test_emits_agents_md_and_claude_shim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = list(generate_specific_agents_md_files(temp_dir, False))

            file_names = [name for name, _ in results]
            assert file_names == ["AGENTS.md", "CLAUDE.md"]
            assert all(status == FileOperationStatus.CREATED for _, status in results)

            assert os.path.exists(os.path.join(temp_dir, "AGENTS.md"))
            claude_path = os.path.join(temp_dir, "CLAUDE.md")
            assert os.path.exists(claude_path)
            with open(claude_path) as f:
                assert f.read().strip() == "@AGENTS.md"
            assert not os.path.exists(os.path.join(temp_dir, ".agent"))

    def test_default_does_not_bundle_offline_docs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            list(generate_specific_agents_md_files(temp_dir, False))

            assert not os.path.exists(
                os.path.join(temp_dir, ".uipath", "llms-full.txt")
            )
            # default AGENTS.md must not reference the offline fallback
            with open(os.path.join(temp_dir, "AGENTS.md")) as f:
                assert ".uipath/llms-full.txt" not in f.read()

    def test_with_offline_docs_bundles_llms_full(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            list(
                generate_specific_agents_md_files(
                    temp_dir, False, with_offline_docs=True
                )
            )

            # llms-full.txt is bundled in the wheel; if running from a dev
            # install where the file is missing, the .uipath dir is still
            # created but the file may be absent. When the file is present,
            # AGENTS.md gains a third step pointing at it.
            uipath_dir = os.path.join(temp_dir, ".uipath")
            assert os.path.isdir(uipath_dir)
            if os.path.exists(os.path.join(uipath_dir, "llms-full.txt")):
                with open(os.path.join(temp_dir, "AGENTS.md")) as f:
                    assert ".uipath/llms-full.txt" in f.read()

    def test_skip_existing_with_no_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            agents_path = os.path.join(temp_dir, "AGENTS.md")
            with open(agents_path, "w") as f:
                f.write("user content")

            results = list(generate_specific_agents_md_files(temp_dir, True))

            statuses = {name: status for name, status in results}
            assert statuses["AGENTS.md"] == FileOperationStatus.SKIPPED
            assert statuses["CLAUDE.md"] == FileOperationStatus.CREATED
            with open(agents_path) as f:
                assert f.read() == "user content"

    def test_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            agents_path = os.path.join(temp_dir, "AGENTS.md")
            with open(agents_path, "w") as f:
                f.write("user content")

            results = list(generate_specific_agents_md_files(temp_dir, False))

            statuses = {name: status for name, status in results}
            assert statuses["AGENTS.md"] == FileOperationStatus.UPDATED
            assert statuses["CLAUDE.md"] == FileOperationStatus.CREATED
            with open(agents_path) as f:
                assert f.read() != "user content"
