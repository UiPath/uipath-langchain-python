"""Azure Blob Storage + SQLite hybrid checkpointer.

This module provides AsyncBlobSqliteSaver, a hybrid checkpointer that uses
Azure Blob Storage as the source of truth while maintaining a local SQLite
database as a performance-optimized cache.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
import urllib.parse
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    SerializerProtocol,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .blob_storage import (
    AzureBlobBackend,
    BlobStorageBackend,
    FilesystemBackend,
)

logger = logging.getLogger(__name__)


class AsyncBlobSqliteSaver(AsyncSqliteSaver):
    """Hybrid checkpointer using blob storage as source of truth.

    This class inherits from AsyncSqliteSaver and adds blob storage
    synchronization. It maintains a local SQLite cache for fast reads while
    using blob storage (Azure, S3, filesystem) as the persistent source of truth.

    Designed for UIPath job integration with flat blob storage structure.
    All checkpoint and write data are stored as JSON files in blob storage
    under a job-specific folder: `uipath_job_{guid}/`

    Key Features:
    - Blob storage as source of truth (survives container restarts)
    - SQLite as local cache (ephemeral, fast reads)
    - Write-through strategy (immediate sync to both)
    - Sync from blob storage on startup (optional)
    - Flat blob storage structure for easy management
    - Optional gzip compression (can reduce size by 60-80%)
    - Multiple backend support (Azure, filesystem, S3)

    Attributes:
        storage_backend: Blob storage backend (Azure, filesystem, etc.)
        job_guid: UIPath job GUID (folder name: uipath_job_{guid})
        sync_on_startup: Download all blobs on startup
        compress: Enable gzip compression for blob storage

    Example:
        >>> # Azure backend
        >>> async with AsyncBlobSqliteSaver.from_azure_blob(
        ...     sqlite_path="/tmp/checkpoint_cache.db",
        ...     connection_string=azure_conn_str,
        ...     container_name="langgraph-checkpoints",
        ...     job_guid="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        ...     sync_on_startup=True,
        ... ) as checkpointer:
        ...     graph = builder.compile(checkpointer=checkpointer)
        ...     result = await graph.ainvoke(input_data, config)

        >>> # Filesystem backend
        >>> async with AsyncBlobSqliteSaver.from_filesystem(
        ...     sqlite_path="/tmp/checkpoint_cache.db",
        ...     storage_path="/var/checkpoints",
        ...     job_guid="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        ... ) as checkpointer:
        ...     graph = builder.compile(checkpointer=checkpointer)
        ...     result = await graph.ainvoke(input_data, config)
    """

    storage_backend: BlobStorageBackend
    job_guid: str
    sync_on_startup: bool
    compress: bool

    def __init__(
        self,
        conn: aiosqlite.Connection,
        storage_backend: BlobStorageBackend,
        job_guid: str,
        *,
        serde: SerializerProtocol | None = None,
        sync_on_startup: bool = True,
        compress: bool = False,
    ):
        """Create instance with existing storage backend.

        Args:
            conn: Existing aiosqlite connection
            storage_backend: Blob storage backend (Azure, filesystem, etc.)
            job_guid: UIPath job GUID (creates folder uipath_job_{guid})
            serde: Optional custom serializer
            sync_on_startup: If True, sync from blob storage on startup
            compress: If True, use gzip compression for blob storage (reduces size by 60-80%)
        """
        # Call parent constructor (AsyncSqliteSaver.__init__)
        super().__init__(conn, serde=serde)

        # Add blob storage attributes
        self.storage_backend = storage_backend
        self.job_guid = job_guid
        self.sync_on_startup = sync_on_startup
        self.compress = compress

    @classmethod
    @asynccontextmanager
    async def from_azure_blob(
        cls,
        sqlite_path: str,
        connection_string: str,
        container_name: str,
        job_guid: str,
        *,
        serde: SerializerProtocol | None = None,
        sync_on_startup: bool = True,
        compress: bool = False,
    ) -> AsyncIterator[AsyncBlobSqliteSaver]:
        """Create instance with Azure Blob Storage backend.

        Args:
            sqlite_path: Path to local SQLite database file (e.g., /tmp/checkpoint_cache.db)
            connection_string: Azure Storage connection string
            container_name: Azure Blob container name
            job_guid: UIPath job GUID (creates folder uipath_job_{guid})
            serde: Optional custom serializer
            sync_on_startup: If True, download checkpoints from blob on startup
            compress: If True, use gzip compression for blob storage (reduces size by 60-80%)

        Yields:
            AsyncBlobSqliteSaver instance with Azure backend

        Example:
            >>> async with AsyncBlobSqliteSaver.from_azure_blob(
            ...     sqlite_path="/tmp/checkpoint_cache.db",
            ...     connection_string="DefaultEndpointsProtocol=https;...",
            ...     container_name="langgraph-checkpoints",
            ...     job_guid="a1b2c3d4-e5f6-7890",
            ...     sync_on_startup=True,
            ...     compress=True,
            ... ) as saver:
            ...     # Use saver
            ...     pass
        """
        # Create Azure backend
        backend = await AzureBlobBackend.from_connection_string(
            connection_string, container_name
        )

        # Create SQLite connection
        async with aiosqlite.connect(sqlite_path) as conn:
            # Create instance with backend
            saver = cls(
                conn,
                backend,
                job_guid,
                serde=serde,
                sync_on_startup=sync_on_startup,
                compress=compress,
            )
            try:
                yield saver
            finally:
                # Clean up backend
                await backend.close()

    @classmethod
    @asynccontextmanager
    async def from_filesystem(
        cls,
        sqlite_path: str,
        storage_path: str | Path,
        job_guid: str,
        *,
        serde: SerializerProtocol | None = None,
        sync_on_startup: bool = True,
        compress: bool = False,
    ) -> AsyncIterator[AsyncBlobSqliteSaver]:
        """Create instance with filesystem storage backend.

        Args:
            sqlite_path: Path to local SQLite database file (e.g., /tmp/checkpoint_cache.db)
            storage_path: Path to directory for storing checkpoint files
            job_guid: UIPath job GUID (creates folder uipath_job_{guid})
            serde: Optional custom serializer
            sync_on_startup: If True, load checkpoints from storage on startup
            compress: If True, use gzip compression for storage (reduces size by 60-80%)

        Yields:
            AsyncBlobSqliteSaver instance with filesystem backend

        Example:
            >>> async with AsyncBlobSqliteSaver.from_filesystem(
            ...     sqlite_path="/tmp/checkpoint_cache.db",
            ...     storage_path="/var/checkpoints",
            ...     job_guid="a1b2c3d4-e5f6-7890",
            ...     sync_on_startup=True,
            ... ) as saver:
            ...     # Use saver
            ...     pass
        """
        # Create filesystem backend
        backend = FilesystemBackend(storage_path)

        # Create SQLite connection
        async with aiosqlite.connect(sqlite_path) as conn:
            # Create instance with backend
            saver = cls(
                conn,
                backend,
                job_guid,
                serde=serde,
                sync_on_startup=sync_on_startup,
                compress=compress,
            )
            try:
                yield saver
            finally:
                # Clean up backend
                await backend.close()

    # =========================================================================
    # Helper Methods: Blob Name Construction and Parsing
    # =========================================================================

    def _get_blob_prefix(self) -> str:
        """Get the blob prefix for this job."""
        return f"uipath_job_{self.job_guid}/"

    def _checkpoint_blob_name(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> str:
        """Construct blob name for a checkpoint.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace (empty string for default)
            checkpoint_id: Checkpoint identifier

        Returns:
            Blob name like: uipath_job_{guid}/checkpoint_{thread_id}_{ns}_{checkpoint_id}.json[.gz]
        """
        # Handle empty checkpoint_ns
        ns = checkpoint_ns if checkpoint_ns else "default"
        # URL-encode parts to handle special characters
        thread_id_safe = urllib.parse.quote(thread_id, safe="")
        ns_safe = urllib.parse.quote(ns, safe="")
        checkpoint_id_safe = urllib.parse.quote(checkpoint_id, safe="")
        extension = ".json.gz" if self.compress else ".json"
        return f"{self._get_blob_prefix()}checkpoint_{thread_id_safe}_{ns_safe}_{checkpoint_id_safe}{extension}"

    def _write_blob_name(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        idx: int,
    ) -> str:
        """Construct blob name for a write.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace (empty string for default)
            checkpoint_id: Checkpoint identifier
            task_id: Task identifier
            idx: Write index

        Returns:
            Blob name like: uipath_job_{guid}/write_{thread_id}_{ns}_{checkpoint_id}_{task_id}_{idx}.json[.gz]
        """
        ns = checkpoint_ns if checkpoint_ns else "default"
        # URL-encode parts
        thread_id_safe = urllib.parse.quote(thread_id, safe="")
        ns_safe = urllib.parse.quote(ns, safe="")
        checkpoint_id_safe = urllib.parse.quote(checkpoint_id, safe="")
        task_id_safe = urllib.parse.quote(task_id, safe="")
        extension = ".json.gz" if self.compress else ".json"
        return f"{self._get_blob_prefix()}write_{thread_id_safe}_{ns_safe}_{checkpoint_id_safe}_{task_id_safe}_{idx}{extension}"

    def _parse_checkpoint_blob_name(
        self, blob_name: str
    ) -> tuple[str, str, str] | None:
        """Parse checkpoint blob name into components.

        Args:
            blob_name: Full blob name

        Returns:
            (thread_id, checkpoint_ns, checkpoint_id) or None if invalid
        """
        # Remove prefix
        name = blob_name.removeprefix(self._get_blob_prefix())
        if not name.startswith("checkpoint_"):
            return None

        # Remove checkpoint_ prefix and .json or .json.gz suffix
        name = name.removeprefix("checkpoint_")
        name = name.removesuffix(".json.gz")
        name = name.removesuffix(".json")

        # Split and decode
        parts = name.split("_")
        if len(parts) != 3:
            return None

        thread_id = urllib.parse.unquote(parts[0])
        checkpoint_ns = urllib.parse.unquote(parts[1])
        checkpoint_id = urllib.parse.unquote(parts[2])

        # Convert "default" back to empty string
        if checkpoint_ns == "default":
            checkpoint_ns = ""

        return (thread_id, checkpoint_ns, checkpoint_id)

    def _parse_write_blob_name(
        self, blob_name: str
    ) -> tuple[str, str, str, str, int] | None:
        """Parse write blob name into components.

        Args:
            blob_name: Full blob name

        Returns:
            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) or None if invalid
        """
        # Remove prefix
        name = blob_name.removeprefix(self._get_blob_prefix())
        if not name.startswith("write_"):
            return None

        # Remove write_ prefix and .json or .json.gz suffix
        name = name.removeprefix("write_")
        name = name.removesuffix(".json.gz")
        name = name.removesuffix(".json")

        # Split and decode
        parts = name.split("_")
        if len(parts) != 5:
            return None

        thread_id = urllib.parse.unquote(parts[0])
        checkpoint_ns = urllib.parse.unquote(parts[1])
        checkpoint_id = urllib.parse.unquote(parts[2])
        task_id = urllib.parse.unquote(parts[3])
        try:
            idx = int(parts[4])
        except ValueError:
            return None

        # Convert "default" back to empty string
        if checkpoint_ns == "default":
            checkpoint_ns = ""

        return (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)

    # =========================================================================
    # Helper Methods: Serialization
    # =========================================================================

    def _serialize_checkpoint(
        self,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        thread_id: str,
        checkpoint_ns: str,
        parent_checkpoint_id: str | None,
    ) -> str:
        """Serialize checkpoint to JSON format for blob storage.

        Args:
            checkpoint: Checkpoint object
            metadata: Checkpoint metadata
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace
            parent_checkpoint_id: Parent checkpoint ID (if any)

        Returns:
            JSON string representation
        """
        # Serialize checkpoint using parent's serde
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)

        # Base64 encode the serialized checkpoint
        checkpoint_b64 = base64.b64encode(serialized_checkpoint).decode("utf-8")

        # Create JSON structure
        data = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint["id"],
            "parent_checkpoint_id": parent_checkpoint_id,
            "type": type_,
            "checkpoint": checkpoint_b64,
            "metadata": metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return json.dumps(data, ensure_ascii=False)

    def _serialize_write(
        self,
        channel: str,
        value: Any,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        idx: int,
    ) -> str:
        """Serialize write to JSON format for blob storage.

        Args:
            channel: Channel name
            value: Value to write
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace
            checkpoint_id: Checkpoint identifier
            task_id: Task identifier
            idx: Write index

        Returns:
            JSON string representation
        """
        # Serialize value using parent's serde
        type_, serialized_value = self.serde.dumps_typed(value)

        # Base64 encode the serialized value
        value_b64 = base64.b64encode(serialized_value).decode("utf-8")

        # Create JSON structure
        data = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "idx": idx,
            "channel": channel,
            "type": type_,
            "value": value_b64,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return json.dumps(data, ensure_ascii=False)

    # =========================================================================
    # Helper Methods: Compression
    # =========================================================================

    def _compress_data(self, data: str) -> bytes:
        """Compress JSON string data using gzip.

        Args:
            data: JSON string to compress

        Returns:
            Compressed bytes
        """
        return gzip.compress(data.encode("utf-8"), compresslevel=6)

    def _decompress_data(self, data: bytes) -> str:
        """Decompress gzip data to JSON string.

        Args:
            data: Compressed bytes

        Returns:
            Decompressed JSON string
        """
        return gzip.decompress(data).decode("utf-8")

    # =========================================================================
    # Helper Methods: Blob Storage Operations
    # =========================================================================

    async def _upload_blob(self, blob_name: str, data: str) -> None:
        """Upload data to blob storage.

        Args:
            blob_name: Name of the blob
            data: JSON string data to upload
        """
        try:
            # Compress data if compression is enabled
            if self.compress:
                upload_data = self._compress_data(data)
            else:
                upload_data = data

            # Upload using the storage backend
            await self.storage_backend.upload(blob_name, upload_data)
            logger.debug(f"Uploaded blob: {blob_name} (compressed={self.compress})")
        except Exception as e:
            logger.error(f"Failed to upload blob {blob_name}: {e}")
            raise

    async def _download_blob(self, blob_name: str) -> str:
        """Download data from blob storage.

        Args:
            blob_name: Name of the blob

        Returns:
            JSON string data
        """
        try:
            # Download using the storage backend
            data = await self.storage_backend.download(blob_name)

            # Detect if the blob is compressed by checking the extension
            is_compressed = blob_name.endswith(".gz")

            if is_compressed:
                # Decompress the data
                return self._decompress_data(data)
            else:
                # Return as-is (plain JSON)
                return data.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to download blob {blob_name}: {e}")
            raise

    async def _sync_from_blob_storage(self) -> None:
        """Download all checkpoints and writes from blob storage.

        This method is called during setup if sync_on_startup is True.
        It downloads all blobs with the job-specific prefix and populates
        the local SQLite cache.
        """
        logger.info(f"Syncing checkpoints from blob storage for job {self.job_guid}")

        prefix = self._get_blob_prefix()
        checkpoint_blobs = []
        write_blobs = []

        try:
            # List all blobs with the job prefix using storage backend
            async for blob_name in self.storage_backend.list_blobs(prefix):
                if blob_name.startswith(f"{prefix}checkpoint_"):
                    checkpoint_blobs.append(blob_name)
                elif blob_name.startswith(f"{prefix}write_"):
                    write_blobs.append(blob_name)

            logger.info(
                f"Found {len(checkpoint_blobs)} checkpoints and {len(write_blobs)} writes"
            )

            # Download and process checkpoints first
            for blob_name in checkpoint_blobs:
                try:
                    data = await self._download_blob(blob_name)
                    checkpoint_data = json.loads(data)

                    # Decode base64 checkpoint
                    checkpoint_bytes = base64.b64decode(checkpoint_data["checkpoint"])

                    # Serialize metadata
                    serialized_metadata = json.dumps(
                        checkpoint_data["metadata"], ensure_ascii=False
                    ).encode("utf-8", "ignore")

                    # Insert into SQLite
                    async with self.conn.execute(
                        "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            checkpoint_data["thread_id"],
                            checkpoint_data["checkpoint_ns"],
                            checkpoint_data["checkpoint_id"],
                            checkpoint_data["parent_checkpoint_id"],
                            checkpoint_data["type"],
                            checkpoint_bytes,
                            serialized_metadata,
                        ),
                    ):
                        pass

                except Exception as e:
                    logger.error(f"Failed to sync checkpoint {blob_name}: {e}")
                    # Continue with other checkpoints

            # Commit checkpoints
            await self.conn.commit()

            # Download and process writes
            for blob_name in write_blobs:
                try:
                    data = await self._download_blob(blob_name)
                    write_data = json.loads(data)

                    # Decode base64 value
                    value_bytes = base64.b64decode(write_data["value"])

                    # Insert into SQLite
                    async with self.conn.execute(
                        "INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            write_data["thread_id"],
                            write_data["checkpoint_ns"],
                            write_data["checkpoint_id"],
                            write_data["task_id"],
                            write_data["idx"],
                            write_data["channel"],
                            write_data["type"],
                            value_bytes,
                        ),
                    ):
                        pass

                except Exception as e:
                    logger.error(f"Failed to sync write {blob_name}: {e}")
                    # Continue with other writes

            # Commit writes
            await self.conn.commit()

            logger.info("Successfully synced from blob storage")

        except Exception as e:
            logger.error(f"Failed to sync from blob storage: {e}")
            raise

    # =========================================================================
    # Overridden Methods: Setup and Write Operations
    # =========================================================================

    async def setup(self) -> None:
        """Set up SQLite tables and optionally sync from blob storage.

        This method calls the parent setup to create SQLite tables, then
        optionally downloads all blobs and populates the SQLite cache if
        sync_on_startup is True.
        """
        # Call parent setup to create SQLite tables
        await super().setup()

        # If sync_on_startup, download all blobs and populate SQLite
        if self.sync_on_startup:
            await self._sync_from_blob_storage()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save checkpoint to SQLite and blob storage immediately.

        This method first calls the parent to write to SQLite, then
        immediately uploads to blob storage (synchronous/blocking upload).

        Args:
            config: The config to associate with the checkpoint
            checkpoint: The checkpoint to save
            metadata: Additional metadata to save with the checkpoint
            new_versions: New channel versions as of this write

        Returns:
            Updated configuration after storing the checkpoint
        """
        # Call parent to write to SQLite
        config = await super().aput(config, checkpoint, metadata, new_versions)

        # Upload to blob storage
        try:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"]["checkpoint_ns"]
            checkpoint_id = checkpoint["id"]
            parent_checkpoint_id = config["configurable"].get("checkpoint_id")

            blob_name = self._checkpoint_blob_name(
                thread_id, checkpoint_ns, checkpoint_id
            )
            json_data = self._serialize_checkpoint(
                checkpoint, metadata, thread_id, checkpoint_ns, parent_checkpoint_id
            )
            await self._upload_blob(blob_name, json_data)
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to blob: {e}")
            # Continue - SQLite has the data, can retry manually if needed

        return config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save writes to SQLite and blob storage immediately.

        This method first calls the parent to write to SQLite, then
        immediately uploads all writes to blob storage in parallel.

        Args:
            config: Configuration of the related checkpoint
            writes: List of writes to store, each as (channel, value) pair
            task_id: Identifier for the task creating the writes
            task_path: Path of the task creating the writes
        """
        # Call parent to write to SQLite
        await super().aput_writes(config, writes, task_id, task_path)

        # Upload each write to blob storage (can parallelize)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        upload_tasks = []
        for idx, (channel, value) in enumerate(writes):
            blob_name = self._write_blob_name(
                thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
            json_data = self._serialize_write(
                channel, value, thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
            upload_tasks.append(self._upload_blob(blob_name, json_data))

        # Wait for all uploads to complete
        try:
            await asyncio.gather(*upload_tasks, return_exceptions=False)
        except Exception as e:
            logger.error(f"Failed to upload writes to blob: {e}")
            # Continue - SQLite has the data

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete thread from SQLite and blob storage.

        This method first calls the parent to delete from SQLite, then
        deletes all blobs for this thread from blob storage.

        Args:
            thread_id: The thread ID to delete
        """
        # Call parent to delete from SQLite
        await super().adelete_thread(thread_id)

        # Delete all blobs for this thread
        try:
            # Construct prefixes for listing blobs
            thread_id_safe = urllib.parse.quote(thread_id, safe="")
            checkpoint_prefix = f"{self._get_blob_prefix()}checkpoint_{thread_id_safe}_"
            write_prefix = f"{self._get_blob_prefix()}write_{thread_id_safe}_"

            # List all blobs matching the prefixes using storage backend
            blobs_to_delete = []

            async for blob_name in self.storage_backend.list_blobs(checkpoint_prefix):
                blobs_to_delete.append(blob_name)

            async for blob_name in self.storage_backend.list_blobs(write_prefix):
                blobs_to_delete.append(blob_name)

            # Delete all matching blobs in parallel
            delete_tasks = []
            for blob_name in blobs_to_delete:
                delete_tasks.append(self.storage_backend.delete(blob_name))

            if delete_tasks:
                await asyncio.gather(*delete_tasks, return_exceptions=True)
                logger.info(
                    f"Deleted {len(blobs_to_delete)} blobs for thread {thread_id}"
                )

        except Exception as e:
            logger.error(f"Failed to delete blobs for thread {thread_id}: {e}")
            # Continue - SQLite already deleted
