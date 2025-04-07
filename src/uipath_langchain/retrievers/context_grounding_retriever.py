from typing import List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from uipath_sdk import UiPathSDK


class ContextGroundingRetriever(BaseRetriever):
    index_name: str
    uipath_sdk: Optional[UiPathSDK] = None
    number_of_results: Optional[int] = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever calls context_grounding API to search the requested index."""

        sdk = self.uipath_sdk if self.uipath_sdk is not None else UiPathSDK()
        results = sdk.context_grounding.search(
            self.index_name,
            query,
            self.number_of_results if self.number_of_results is not None else 10,
        )

        return [
            Document(
                page_content=x.content,
                metadata={
                    "source": x.source,
                    "page_number": x.page_number,
                },
            )
            for x in results
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async implementations for retriever calls context_grounding API to search the requested index."""

        sdk = self.uipath_sdk if self.uipath_sdk is not None else UiPathSDK()
        results = await sdk.context_grounding.search_async(
            self.index_name,
            query,
            self.number_of_results if self.number_of_results is not None else 10,
        )

        return [
            Document(
                page_content=x.content,
                metadata={
                    "source": x.source,
                    "page_number": x.page_number,
                },
            )
            for x in results
        ]
