"""UiPath Buckets Backend for Deep Agents.

Backend implementation using UiPath Storage Buckets via the UiPath Python SDK.
"""

from __future__ import annotations

import fnmatch
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

import wcmatch.glob as wcglob
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.utils import (
    check_empty_content,
    format_content_with_line_numbers,
    perform_string_replacement,
)
from uipath.platform import UiPath
from uipath.platform.common import PagedResult
from uipath.platform.orchestrator import BucketFile

__all__ = ["UiPathBucketBackend", "UiPathBucketConfig"]


@dataclass
class UiPathBucketConfig:
    """Configuration for UiPath Bucket backend.

    Attributes:
        bucket_name: Name of the UiPath storage bucket.
        folder_path: UiPath folder path where the bucket resides.
        prefix: Optional prefix for all file paths within the bucket.
    """

    bucket_name: str
    folder_path: str | None = None
    prefix: str = ""


class UiPathBucketBackend(BackendProtocol):
    """UiPath Storage Buckets backend for Deep Agents file operations.

    Uses the UiPath Python SDK to interact with UiPath Storage Buckets.
    Files are stored with content as JSON: {"content": [...lines], "created_at": "...", "modified_at": "..."}

    Example:
        >>> config = UiPathBucketConfig(bucket_name="my-agent-storage", folder_name="MyFolder")
        >>> backend = UiPathBucketBackend(config)
    """

    def __init__(self, config: UiPathBucketConfig) -> None:
        """Initialize the UiPath Bucket backend.

        Args:
            config: Bucket configuration.
        """
        self._sdk = UiPath()
        self._config = config
        self._prefix = config.prefix.strip("/")
        if self._prefix:
            self._prefix += "/"

    def _blob_path(self, path: str) -> str:
        """Convert virtual path to blob file path in bucket."""
        clean = path.lstrip("/")
        return f"{self._prefix}{clean}"

    def _virtual_path(self, blob_path: str) -> str:
        """Convert blob file path to virtual path."""
        if self._prefix and blob_path.startswith(self._prefix):
            blob_path = blob_path[len(self._prefix) :]
        return "/" + blob_path.lstrip("/")

    def _bucket_kwargs(self) -> dict[str, Any]:
        """Get common bucket identification kwargs."""
        kwargs: dict[str, Any] = {"name": self._config.bucket_name}
        if self._config.folder_path:
            kwargs["folder_path"] = self._config.folder_path
        return kwargs

    def _download_content(self, path: str) -> bytes | None:
        """Download file content from bucket."""
        blob_path = self._blob_path(path)
        try:
            if not self._sdk.buckets.exists_file(
                blob_file_path=blob_path,
                **self._bucket_kwargs(),
            ):
                return None

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name

            self._sdk.buckets.download(
                blob_file_path=blob_path,
                destination_path=tmp_path,
                **self._bucket_kwargs(),
            )

            with open(tmp_path, "rb") as f:
                return f.read()
        except Exception:
            return None

    async def _adownload_content(self, path: str) -> bytes | None:
        """Download file content from bucket asynchronously."""
        blob_path = self._blob_path(path)
        try:
            if not await self._sdk.buckets.exists_file_async(
                blob_file_path=blob_path,
                **self._bucket_kwargs(),
            ):
                return None

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name

            await self._sdk.buckets.download_async(
                blob_file_path=blob_path,
                destination_path=tmp_path,
                **self._bucket_kwargs(),
            )

            with open(tmp_path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def _get_file_data(self, path: str) -> dict[str, Any] | None:
        """Get file data dict from bucket."""
        content = self._download_content(path)
        if content is None:
            return None
        try:
            return json.loads(content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            lines = content.decode("utf-8", errors="replace").splitlines()
            return {"content": lines}

    async def _aget_file_data(self, path: str) -> dict[str, Any] | None:
        """Get file data dict from bucket asynchronously."""
        content = await self._adownload_content(path)
        if content is None:
            return None
        try:
            return json.loads(content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            lines = content.decode("utf-8", errors="replace").splitlines()
            return {"content": lines}

    def _put_file_data(
        self, path: str, data: dict[str, Any], *, update_modified: bool = True
    ) -> None:
        """Upload file data dict to bucket."""
        blob_path = self._blob_path(path)
        if update_modified:
            data["modified_at"] = datetime.now(timezone.utc).isoformat()
        content = json.dumps(data).encode("utf-8")

        self._sdk.buckets.upload(
            blob_file_path=blob_path,
            content=content,
            content_type="application/json",
            **self._bucket_kwargs(),
        )

    async def _aput_file_data(
        self, path: str, data: dict[str, Any], *, update_modified: bool = True
    ) -> None:
        """Upload file data dict to bucket asynchronously."""
        blob_path = self._blob_path(path)
        if update_modified:
            data["modified_at"] = datetime.now(timezone.utc).isoformat()
        content = json.dumps(data).encode("utf-8")

        await self._sdk.buckets.upload_async(
            blob_file_path=blob_path,
            content=content,
            content_type="application/json",
            **self._bucket_kwargs(),
        )

    def _exists(self, path: str) -> bool:
        """Check if file exists in bucket."""
        blob_path = self._blob_path(path)
        try:
            return self._sdk.buckets.exists_file(
                blob_file_path=blob_path,
                **self._bucket_kwargs(),
            )
        except Exception:
            return False

    async def _aexists(self, path: str) -> bool:
        """Check if file exists in bucket asynchronously."""
        blob_path = self._blob_path(path)
        try:
            return await self._sdk.buckets.exists_file_async(
                blob_file_path=blob_path,
                **self._bucket_kwargs(),
            )
        except Exception:
            return False

    def _list_files(self, prefix: str = "") -> list[BucketFile]:
        """List all files with a prefix."""
        full_prefix = self._blob_path(prefix) if prefix else self._prefix
        full_prefix = full_prefix.rstrip("/")
        results: list[BucketFile] = []

        token: str | None = None
        while True:
            page: PagedResult[BucketFile] = self._sdk.buckets.list_files(
                prefix=full_prefix,
                continuation_token=token,
                **self._bucket_kwargs(),
            )
            results.extend(page.items)
            if not page.continuation_token:
                break
            token = page.continuation_token

        return results

    async def _alist_files(self, prefix: str = "") -> list[BucketFile]:
        """List all files with a prefix asynchronously."""
        full_prefix = self._blob_path(prefix) if prefix else self._prefix
        full_prefix = full_prefix.rstrip("/")
        results: list[BucketFile] = []

        token: str | None = None
        while True:
            page: PagedResult[BucketFile] = await self._sdk.buckets.list_files_async(
                prefix=full_prefix,
                continuation_token=token,
                **self._bucket_kwargs(),
            )
            results.extend(page.items)
            if not page.continuation_token:
                break
            token = page.continuation_token

        return results

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files in a directory."""
        prefix = path.lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        files = self._list_files(prefix)
        results: list[FileInfo] = []
        seen_dirs: set[str] = set()

        for file in files:
            vpath = self._virtual_path(file.path)

            rel = vpath[len("/" + prefix) :] if prefix else vpath[1:]
            if "/" in rel:
                dir_name = rel.split("/")[0]
                dir_path = "/" + prefix + dir_name + "/"
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    results.append({"path": dir_path, "is_dir": True})
            else:
                results.append({
                    "path": vpath,
                    "is_dir": False,
                    "size": file.size or 0,
                    "modified_at": file.last_modified,
                })

        results.sort(key=lambda x: x.get("path", ""))
        return results

    async def als_info(self, path: str) -> list[FileInfo]:
        """List files in a directory asynchronously."""
        prefix = path.lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        files = await self._alist_files(prefix)
        results: list[FileInfo] = []
        seen_dirs: set[str] = set()

        for file in files:
            vpath = self._virtual_path(file.path)

            rel = vpath[len("/" + prefix) :] if prefix else vpath[1:]
            if "/" in rel:
                dir_name = rel.split("/")[0]
                dir_path = "/" + prefix + dir_name + "/"
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    results.append({"path": dir_path, "is_dir": True})
            else:
                results.append({
                    "path": vpath,
                    "is_dir": False,
                    "size": file.size or 0,
                    "modified_at": file.last_modified,
                })

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content with line numbers."""
        data = self._get_file_data(file_path)
        if data is None:
            return f"Error: File '{file_path}' not found"

        lines = data.get("content", [])
        if not lines:
            empty_msg = check_empty_content("")
            if empty_msg:
                return empty_msg

        if offset >= len(lines):
            return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

        selected = lines[offset : offset + limit]
        return format_content_with_line_numbers(selected, start_line=offset + 1)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content with line numbers asynchronously."""
        data = await self._aget_file_data(file_path)
        if data is None:
            return f"Error: File '{file_path}' not found"

        lines = data.get("content", [])
        if not lines:
            empty_msg = check_empty_content("")
            if empty_msg:
                return empty_msg

        if offset >= len(lines):
            return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

        selected = lines[offset : offset + limit]
        return format_content_with_line_numbers(selected, start_line=offset + 1)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create a new file."""
        if self._exists(file_path):
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. "
                "Read and then make an edit, or write to a new path."
            )

        now = datetime.now(timezone.utc).isoformat()
        data = {
            "content": content.splitlines(),
            "created_at": now,
            "modified_at": now,
        }
        try:
            self._put_file_data(file_path, data, update_modified=False)
            return WriteResult(path=file_path, files_update=None)
        except Exception as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Create a new file asynchronously."""
        if await self._aexists(file_path):
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. "
                "Read and then make an edit, or write to a new path."
            )

        now = datetime.now(timezone.utc).isoformat()
        data = {
            "content": content.splitlines(),
            "created_at": now,
            "modified_at": now,
        }
        try:
            await self._aput_file_data(file_path, data, update_modified=False)
            return WriteResult(path=file_path, files_update=None)
        except Exception as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit file by replacing strings."""
        data = self._get_file_data(file_path)
        if data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = "\n".join(data.get("content", []))
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        data["content"] = new_content.splitlines()

        try:
            self._put_file_data(file_path, data)
            return EditResult(
                path=file_path, files_update=None, occurrences=int(occurrences)
            )
        except Exception as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit file by replacing strings asynchronously."""
        data = await self._aget_file_data(file_path)
        if data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = "\n".join(data.get("content", []))
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        data["content"] = new_content.splitlines()

        try:
            await self._aput_file_data(file_path, data)
            return EditResult(
                path=file_path, files_update=None, occurrences=int(occurrences)
            )
        except Exception as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

    def grep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> list[GrepMatch] | str:
        """Search for pattern in files."""
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        search_prefix = (path or "/").lstrip("/")
        files = self._list_files(search_prefix)
        matches: list[GrepMatch] = []

        for file in files:
            vpath = self._virtual_path(file.path)
            filename = PurePosixPath(vpath).name

            if glob and not wcglob.globmatch(filename, glob, flags=wcglob.BRACE):
                continue

            data = self._get_file_data(vpath)
            if data is None:
                continue

            for line_num, line in enumerate(data.get("content", []), 1):
                if regex.search(line):
                    matches.append({"path": vpath, "line": line_num, "text": line})

        return matches

    async def agrep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> list[GrepMatch] | str:
        """Search for pattern in files asynchronously."""
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        search_prefix = (path or "/").lstrip("/")
        files = await self._alist_files(search_prefix)
        matches: list[GrepMatch] = []

        for file in files:
            vpath = self._virtual_path(file.path)
            filename = PurePosixPath(vpath).name

            if glob and not wcglob.globmatch(filename, glob, flags=wcglob.BRACE):
                continue

            data = await self._aget_file_data(vpath)
            if data is None:
                continue

            for line_num, line in enumerate(data.get("content", []), 1):
                if regex.search(line):
                    matches.append({"path": vpath, "line": line_num, "text": line})

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob pattern."""
        search_prefix = path.lstrip("/")
        files = self._list_files(search_prefix)
        results: list[FileInfo] = []

        for file in files:
            vpath = self._virtual_path(file.path)
            rel_path = vpath[len(path) :].lstrip("/") if path != "/" else vpath[1:]

            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(vpath, pattern):
                results.append({
                    "path": vpath,
                    "is_dir": False,
                    "size": file.size or 0,
                    "modified_at": file.last_modified,
                })

        results.sort(key=lambda x: x.get("path", ""))
        return results

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob pattern asynchronously."""
        search_prefix = path.lstrip("/")
        files = await self._alist_files(search_prefix)
        results: list[FileInfo] = []

        for file in files:
            vpath = self._virtual_path(file.path)
            rel_path = vpath[len(path) :].lstrip("/") if path != "/" else vpath[1:]

            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(vpath, pattern):
                results.append({
                    "path": vpath,
                    "is_dir": False,
                    "size": file.size or 0,
                    "modified_at": file.last_modified,
                })

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files."""
        responses: list[FileUploadResponse] = []

        for path, content in files:
            try:
                blob_path = self._blob_path(path)
                self._sdk.buckets.upload(
                    blob_file_path=blob_path,
                    content=content,
                    **self._bucket_kwargs(),
                )
                responses.append(FileUploadResponse(path=path, error=None))
            except LookupError:
                responses.append(FileUploadResponse(path=path, error="file_not_found"))
            except PermissionError:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
            except Exception:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))

        return responses

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload multiple files asynchronously."""
        responses: list[FileUploadResponse] = []

        for path, content in files:
            try:
                blob_path = self._blob_path(path)
                await self._sdk.buckets.upload_async(
                    blob_file_path=blob_path,
                    content=content,
                    **self._bucket_kwargs(),
                )
                responses.append(FileUploadResponse(path=path, error=None))
            except LookupError:
                responses.append(FileUploadResponse(path=path, error="file_not_found"))
            except PermissionError:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
            except Exception:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files."""
        responses: list[FileDownloadResponse] = []

        for path in paths:
            try:
                content = self._download_content(path)
                if content is None:
                    responses.append(
                        FileDownloadResponse(path=path, content=None, error="file_not_found")
                    )
                else:
                    responses.append(
                        FileDownloadResponse(path=path, content=content, error=None)
                    )
            except PermissionError:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="permission_denied")
                )
            except Exception:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )

        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files asynchronously."""
        responses: list[FileDownloadResponse] = []

        for path in paths:
            try:
                content = await self._adownload_content(path)
                if content is None:
                    responses.append(
                        FileDownloadResponse(path=path, content=None, error="file_not_found")
                    )
                else:
                    responses.append(
                        FileDownloadResponse(path=path, content=content, error=None)
                    )
            except PermissionError:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="permission_denied")
                )
            except Exception:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )

        return responses
