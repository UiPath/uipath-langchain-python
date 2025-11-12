"""Blob storage backend abstraction and implementations.

This module provides an abstraction layer for different blob storage backends,
allowing the checkpointer to work with Azure Blob Storage, filesystem, S3, etc.
"""

from __future__ import annotations

import gzip
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator


class BlobStorageBackend(ABC):
    """Abstract base class for blob storage backends.

    This defines the interface that all storage backends must implement.
    Backends can be Azure Blob Storage, S3, filesystem, etc.
    """

    @abstractmethod
    async def upload(self, blob_name: str, data: bytes | str) -> None:
        """Upload data to storage.

        Args:
            blob_name: Name/path of the blob
            data: Data to upload (bytes or string)
        """
        pass

    @abstractmethod
    async def download(self, blob_name: str) -> bytes:
        """Download data from storage.

        Args:
            blob_name: Name/path of the blob

        Returns:
            Downloaded data as bytes
        """
        pass

    @abstractmethod
    async def list_blobs(self, prefix: str) -> AsyncIterator[str]:
        """List all blobs with the given prefix.

        Args:
            prefix: Prefix to filter blobs

        Yields:
            Blob names matching the prefix
        """
        pass

    @abstractmethod
    async def delete(self, blob_name: str) -> None:
        """Delete a blob from storage.

        Args:
            blob_name: Name/path of the blob to delete
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close/cleanup the storage backend."""
        pass


class AzureBlobBackend(BlobStorageBackend):
    """Azure Blob Storage backend implementation."""

    def __init__(self, container_client):
        """Initialize Azure backend.

        Args:
            container_client: Azure ContainerClient instance
        """
        from azure.storage.blob.aio import ContainerClient

        self.container_client: ContainerClient = container_client

    @classmethod
    async def from_connection_string(
        cls, connection_string: str, container_name: str
    ) -> AzureBlobBackend:
        """Create backend from Azure connection string.

        Args:
            connection_string: Azure Storage connection string
            container_name: Container name

        Returns:
            AzureBlobBackend instance
        """
        from azure.storage.blob.aio import BlobServiceClient

        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        container_client = blob_service_client.get_container_client(container_name)
        backend = cls(container_client)
        backend._blob_service_client = blob_service_client  # Store for cleanup
        return backend

    async def upload(self, blob_name: str, data: bytes | str) -> None:
        """Upload data to Azure Blob Storage."""
        blob_client = self.container_client.get_blob_client(blob_name)

        # Determine content type
        if isinstance(data, str):
            upload_data = data
            content_type = (
                "application/gzip" if blob_name.endswith(".gz") else "application/json"
            )
        else:
            upload_data = data
            content_type = (
                "application/gzip" if blob_name.endswith(".gz") else "application/json"
            )

        await blob_client.upload_blob(
            upload_data,
            overwrite=True,
            content_settings={"content_type": content_type},
        )

    async def download(self, blob_name: str) -> bytes:
        """Download data from Azure Blob Storage."""
        blob_client = self.container_client.get_blob_client(blob_name)
        downloader = await blob_client.download_blob()
        return await downloader.readall()

    async def list_blobs(self, prefix: str) -> AsyncIterator[str]:
        """List blobs with given prefix from Azure."""
        async for blob in self.container_client.list_blobs(name_starts_with=prefix):
            yield blob.name

    async def delete(self, blob_name: str) -> None:
        """Delete blob from Azure Blob Storage."""
        blob_client = self.container_client.get_blob_client(blob_name)
        await blob_client.delete_blob()

    async def close(self) -> None:
        """Close Azure client."""
        if hasattr(self, "_blob_service_client"):
            await self._blob_service_client.close()


class FilesystemBackend(BlobStorageBackend):
    """Filesystem-based storage backend implementation.

    Stores blobs as files in a local directory structure.
    """

    def __init__(self, base_path: str | Path):
        """Initialize filesystem backend.

        Args:
            base_path: Base directory path for storing blobs
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def upload(self, blob_name: str, data: bytes | str) -> None:
        """Upload data to filesystem."""
        file_path = self.base_path / blob_name

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write data
        if isinstance(data, str):
            file_path.write_text(data, encoding="utf-8")
        else:
            file_path.write_bytes(data)

    async def download(self, blob_name: str) -> bytes:
        """Download data from filesystem."""
        file_path = self.base_path / blob_name
        return file_path.read_bytes()

    async def list_blobs(self, prefix: str) -> AsyncIterator[str]:
        """List files with given prefix from filesystem."""
        # Convert prefix to path pattern
        prefix_path = self.base_path / prefix

        # Get all files under the prefix path
        if prefix_path.is_dir():
            for file_path in prefix_path.rglob("*"):
                if file_path.is_file():
                    # Return relative path from base_path
                    relative_path = file_path.relative_to(self.base_path)
                    yield str(relative_path).replace(os.sep, "/")
        elif prefix_path.parent.is_dir():
            # Prefix is a partial filename, list matching files
            for file_path in prefix_path.parent.glob(f"{prefix_path.name}*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.base_path)
                    yield str(relative_path).replace(os.sep, "/")

    async def delete(self, blob_name: str) -> None:
        """Delete file from filesystem."""
        file_path = self.base_path / blob_name
        if file_path.exists():
            file_path.unlink()

            # Clean up empty parent directories
            try:
                parent = file_path.parent
                while parent != self.base_path and parent.exists():
                    if not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                    else:
                        break
            except OSError:
                pass  # Directory not empty or other error

    async def close(self) -> None:
        """Close filesystem backend (no-op)."""
        pass
