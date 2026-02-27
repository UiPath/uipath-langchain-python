"""Tests for build_file_content_block in agent/multimodal/invoke.py."""

from unittest.mock import AsyncMock, patch

from uipath_langchain.agent.multimodal import FileInfo
from uipath_langchain.agent.multimodal.invoke import build_file_content_block
from uipath_langchain.agent.multimodal.utils import is_text


class TestIsText:
    """Test cases for the is_text helper."""

    def test_text_plain(self) -> None:
        assert is_text("text/plain") is True

    def test_text_csv(self) -> None:
        assert is_text("text/csv") is True

    def test_text_xml(self) -> None:
        assert is_text("text/xml") is True

    def test_text_html(self) -> None:
        assert is_text("text/html") is True

    def test_text_markdown(self) -> None:
        assert is_text("text/markdown") is True

    def test_text_tab_separated_values(self) -> None:
        assert is_text("text/tab-separated-values") is True

    def test_application_xml(self) -> None:
        assert is_text("application/xml") is True

    def test_application_json(self) -> None:
        assert is_text("application/json") is True

    def test_application_x_yaml(self) -> None:
        assert is_text("application/x-yaml") is True

    def test_case_insensitive(self) -> None:
        assert is_text("TEXT/CSV") is True
        assert is_text("Text/Plain") is True

    def test_image_not_text(self) -> None:
        assert is_text("image/png") is False

    def test_pdf_not_text(self) -> None:
        assert is_text("application/pdf") is False

    def test_unknown_not_text(self) -> None:
        assert is_text("application/octet-stream") is False


class TestBuildFileContentBlockText:
    """Test cases for text MIME type handling in build_file_content_block."""

    @patch(
        "uipath_langchain.agent.multimodal.invoke.download_file_text",
        new_callable=AsyncMock,
    )
    async def test_text_csv_returns_plaintext_block(
        self, mock_download: AsyncMock
    ) -> None:
        """Text/csv files are returned as inline plaintext content blocks."""
        mock_download.return_value = "col1,col2\nval1,val2"

        file_info = FileInfo(
            url="https://example.com/data.csv",
            name="data.csv",
            mime_type="text/csv",
        )
        block = await build_file_content_block(file_info)

        assert block["type"] == "text-plain"
        assert block["text"] == "col1,col2\nval1,val2"
        assert block["title"] == "data.csv"
        mock_download.assert_awaited_once_with("https://example.com/data.csv")

    @patch(
        "uipath_langchain.agent.multimodal.invoke.download_file_text",
        new_callable=AsyncMock,
    )
    async def test_text_plain_returns_plaintext_block(
        self, mock_download: AsyncMock
    ) -> None:
        """Text/plain files are returned as inline plaintext content blocks."""
        mock_download.return_value = "Hello, world!"

        file_info = FileInfo(
            url="https://example.com/readme.txt",
            name="readme.txt",
            mime_type="text/plain",
        )
        block = await build_file_content_block(file_info)

        assert block["type"] == "text-plain"
        assert block["text"] == "Hello, world!"
        assert block["title"] == "readme.txt"
        mock_download.assert_awaited_once_with("https://example.com/readme.txt")

    @patch(
        "uipath_langchain.agent.multimodal.invoke.download_file_text",
        new_callable=AsyncMock,
    )
    async def test_text_xml_returns_plaintext_block(
        self, mock_download: AsyncMock
    ) -> None:
        """Text/xml files (e.g. .xaml) are returned as inline plaintext blocks."""
        xml_content = '<?xml version="1.0"?><root><item>value</item></root>'
        mock_download.return_value = xml_content

        file_info = FileInfo(
            url="https://example.com/workflow.xaml",
            name="workflow.xaml",
            mime_type="text/xml",
        )
        block = await build_file_content_block(file_info)

        assert block["type"] == "text-plain"
        assert block["text"] == xml_content
        assert block["title"] == "workflow.xaml"

    @patch(
        "uipath_langchain.agent.multimodal.invoke.download_file_text",
        new_callable=AsyncMock,
    )
    async def test_application_json_returns_plaintext_block(
        self, mock_download: AsyncMock
    ) -> None:
        """Application/json files are returned as inline plaintext blocks."""
        json_content = '{"key": "value"}'
        mock_download.return_value = json_content

        file_info = FileInfo(
            url="https://example.com/config.json",
            name="config.json",
            mime_type="application/json",
        )
        block = await build_file_content_block(file_info)

        assert block["type"] == "text-plain"
        assert block["text"] == json_content


class TestBuildFileContentBlockImage:
    """Ensure existing image handling still works."""

    @patch(
        "uipath_langchain.agent.multimodal.invoke.download_file_base64",
        new_callable=AsyncMock,
    )
    async def test_image_png(self, mock_download: AsyncMock) -> None:
        mock_download.return_value = "base64encodeddata"

        file_info = FileInfo(
            url="https://example.com/photo.png",
            name="photo.png",
            mime_type="image/png",
        )
        block = await build_file_content_block(file_info)

        assert block["type"] == "image"
        mock_download.assert_awaited_once_with("https://example.com/photo.png")


class TestBuildFileContentBlockPdf:
    """Ensure existing PDF handling still works."""

    @patch(
        "uipath_langchain.agent.multimodal.invoke.download_file_base64",
        new_callable=AsyncMock,
    )
    async def test_pdf(self, mock_download: AsyncMock) -> None:
        mock_download.return_value = "base64pdfdata"

        file_info = FileInfo(
            url="https://example.com/doc.pdf",
            name="doc.pdf",
            mime_type="application/pdf",
        )
        block = await build_file_content_block(file_info)

        assert block["type"] == "file"
        mock_download.assert_awaited_once_with("https://example.com/doc.pdf")


class TestBuildFileContentBlockUnsupported:
    """Test graceful degradation for unsupported MIME types."""

    @patch(
        "uipath_langchain.agent.multimodal.invoke.download_file_base64",
        new_callable=AsyncMock,
    )
    async def test_unsupported_type_returns_plaintext_placeholder(
        self, mock_download: AsyncMock
    ) -> None:
        """Unsupported types return a descriptive plaintext block, not an error."""
        mock_download.return_value = "binarydata"

        file_info = FileInfo(
            url="https://example.com/archive.zip",
            name="archive.zip",
            mime_type="application/zip",
        )
        block = await build_file_content_block(file_info)

        assert block["type"] == "text-plain"
        assert "archive.zip" in block["text"]
        assert "application/zip" in block["text"]
        assert "could not be processed" in block["text"]

    @patch(
        "uipath_langchain.agent.multimodal.invoke.download_file_base64",
        new_callable=AsyncMock,
    )
    async def test_unsupported_type_does_not_raise(
        self, mock_download: AsyncMock
    ) -> None:
        """Unsupported MIME types must not raise ValueError."""
        mock_download.return_value = "data"

        file_info = FileInfo(
            url="https://example.com/model.bin",
            name="model.bin",
            mime_type="application/octet-stream",
        )
        # Should not raise
        block = await build_file_content_block(file_info)
        assert block is not None

    @patch(
        "uipath_langchain.agent.multimodal.invoke.download_file_base64",
        new_callable=AsyncMock,
    )
    async def test_empty_mime_type_returns_placeholder(
        self, mock_download: AsyncMock
    ) -> None:
        """Empty MIME type is treated as unsupported and returns a placeholder."""
        mock_download.return_value = "data"

        file_info = FileInfo(
            url="https://example.com/unknown",
            name="unknown",
            mime_type="",
        )
        block = await build_file_content_block(file_info)

        assert block["type"] == "text-plain"
        assert "could not be processed" in block["text"]
