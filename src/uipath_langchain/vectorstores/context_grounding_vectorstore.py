"""
Vector store implementation that connects to UiPath Context Grounding as a backend.

This is a read-only vector store that uses the UiPath Context Grounding API to retrieve documents.
"""

from collections.abc import Iterable
from typing import Any, Self

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typing_extensions import override
from uipath.platform import UiPath
from uipath.platform.context_grounding import ContextGroundingQueryResponse


class ContextGroundingVectorStore(VectorStore):
    """Vector store that uses UiPath Context Grounding (ECS) as a backend.

    This class provides a straightforward implementation that connects to the
    UiPath Context Grounding API for semantic searching.
    """

    def __init__(
        self,
        index_name: str,
        uipath_sdk: UiPath | None = None,
        folder_path: str | None = None,
    ):
        """Initialize the ContextGroundingVectorStore.

        Args:
            index_name: Name of the context grounding index to use (schema name)
            uipath_sdk: Optional UiPath SDK instance.
            folder_path: Optional folder path for folder-scoped operations
        """
        self.index_name = index_name
        self.folder_path = folder_path
        self.sdk = uipath_sdk or UiPath()

    # VectorStore implementation methods

    @override
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Return documents most similar to the query along with the distances.

        Args:
            query: The query string
            k: Number of results to return (default=4)

        Returns:
            list of tuples of (document, score)
        """
        # Use the context grounding service to perform search
        results: list[ContextGroundingQueryResponse] = (
            self.sdk.context_grounding.search(
                name=self.index_name,
                query=query,
                number_of_results=k,
                folder_path=self.folder_path,
            )
        )

        return self._convert_results_to_documents(results)

    @override
    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Return documents along with their relevance scores on a scale from 0 to 1.

        Args:
            query: The query string
            k: Number of documents to return (default=4)

        Returns:
            list of tuples of (document, relevance_score)
        """
        return [
            (doc, 1.0 - score)
            for doc, score in self.similarity_search_with_score(query, k, **kwargs)
        ]

    @override
    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Asynchronously return documents most similar to the query along with scores.

        Args:
            query: The query string
            k: Number of results to return (default=4)

        Returns:
            list of tuples of (document, score)
        """
        # Use the context grounding service to perform async search
        results: list[
            ContextGroundingQueryResponse
        ] = await self.sdk.context_grounding.search_async(
            name=self.index_name,
            query=query,
            number_of_results=k,
            folder_path=self.folder_path,
        )

        return self._convert_results_to_documents(results)

    @override
    async def asimilarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Asynchronously return documents along with their relevance scores.

        Args:
            query: The query string
            k: Number of documents to return (default=4)

        Returns:
            list of tuples of (document, relevance_score)
        """
        return [
            (doc, 1.0 - score)
            for doc, score in await self.asimilarity_search_with_score(
                query, k, **kwargs
            )
        ]

    @override
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Return documents most similar to the query.

        Args:
            query: The query string
            k: Number of results to return (default=4)

        Returns:
            list of documents most similar to the query
        """
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    @override
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Asynchronously return documents most similar to the query.

        Args:
            query: The query string
            k: Number of results to return (default=4)

        Returns:
            list of documents most similar to the query
        """
        docs_and_scores = await self.asimilarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def _convert_results_to_documents(
        self, results: list[ContextGroundingQueryResponse]
    ) -> list[tuple[Document, float]]:
        """Convert API results to Document objects with scores.

        Args:
            results: List of ContextGroundingQueryResponse objects

        Returns:
            List of tuples containing (Document, score)
        """
        docs_with_scores = []

        for result in results:
            # Create metadata from result fields
            metadata = {}

            # Add string fields with proper defaults
            if result.source:
                metadata["source"] = str(result.source)
            if result.reference:
                metadata["reference"] = str(result.reference)
            if result.page_number:
                metadata["page_number"] = str(result.page_number)
            if result.source_document_id:
                metadata["source_document_id"] = str(result.source_document_id)
            if result.caption:
                metadata["caption"] = str(result.caption)

            # Add any operation metadata if available
            if result.metadata:
                if result.metadata.operation_id:
                    metadata["operation_id"] = str(result.metadata.operation_id)
                if result.metadata.strategy:
                    metadata["strategy"] = str(result.metadata.strategy)

            # Create a Document with the content and metadata
            doc = Document(
                page_content=result.content or "",
                metadata=metadata,
            )

            # Convert score to distance (1 - score)
            score = 1.0 - float(result.score or 0.0)

            docs_with_scores.append((doc, score))

        return docs_with_scores

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Self:
        """This method is required by the VectorStore abstract class, but is not supported
        by ContextGroundingVectorStore which is read-only.

        Raises:
            NotImplementedError: This method is not supported by ContextGroundingVectorStore
        """
        raise NotImplementedError(
            "ContextGroundingVectorStore is a read-only wrapper for UiPath Context Grounding."
        )

    @override
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Not implemented for ContextGroundingVectorStore as this is a read-only wrapper."""
        raise NotImplementedError(
            "ContextGroundingVectorStore is a read-only wrapper for UiPath Context Grounding."
        )

    @override
    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Not implemented for ContextGroundingVectorStore as this is a read-only wrapper."""
        raise NotImplementedError(
            "ContextGroundingVectorStore is a read-only wrapper for UiPath Context Grounding."
        )
