from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from uipath.platform import UiPath
from uipath.platform.context_grounding import UnifiedSearchScope


class ContextGroundingRetriever(BaseRetriever):
    index_name: str
    folder_path: str | None = None
    folder_key: str | None = None
    uipath_sdk: UiPath | None = None
    number_of_results: int | None = 10
    scope_folder: str | None = None
    scope_extension: str | None = None

    def _build_scope(self) -> UnifiedSearchScope | None:
        if self.scope_folder or self.scope_extension:
            return UnifiedSearchScope(
                folder=self.scope_folder,
                extension=self.scope_extension,
            )
        return None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Sync implementation calls context_grounding unified_search API."""

        sdk = self.uipath_sdk if self.uipath_sdk is not None else UiPath()
        result = sdk.context_grounding.unified_search(
            self.index_name,
            query,
            number_of_results=self.number_of_results
            if self.number_of_results is not None
            else 10,
            scope=self._build_scope(),
            folder_path=self.folder_path,
            folder_key=self.folder_key,
        )

        values = result.semantic_results.values if result.semantic_results else []

        return [
            Document(
                page_content=x.content,
                metadata={
                    "source": x.source,
                    "reference": x.reference,
                    "page_number": x.page_number,
                    "score": x.score,
                },
            )
            for x in values
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Async implementation calls context_grounding unified_search_async API."""

        sdk = self.uipath_sdk if self.uipath_sdk is not None else UiPath()
        result = await sdk.context_grounding.unified_search_async(
            self.index_name,
            query,
            number_of_results=self.number_of_results
            if self.number_of_results is not None
            else 10,
            scope=self._build_scope(),
            folder_path=self.folder_path,
            folder_key=self.folder_key,
        )

        values = result.semantic_results.values if result.semantic_results else []

        return [
            Document(
                page_content=x.content,
                metadata={
                    "source": x.source,
                    "reference": x.reference,
                    "page_number": x.page_number,
                    "score": x.score,
                },
            )
            for x in values
        ]
