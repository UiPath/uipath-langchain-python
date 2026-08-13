"""
UiPath Context Grounding service for document knowledge base and search.
Focuses specifically on Context Grounding capabilities using UiPath SDK.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

# UiPath LangChain integration for Context Grounding
from uipath_langchain.retrievers import ContextGroundingRetriever
from uipath_langchain.vectorstores.context_grounding_vectorstore import ContextGroundingVectorStore
from uipath_langchain.chat.models import UiPathAzureChatOpenAI

try:
    from ..config.settings import settings
except ImportError:
    from config.settings import settings

logger = logging.getLogger(__name__)


class ContextGroundingError(Exception):
    """Custom exception for context grounding errors."""
    pass


class IndexConfig:
    """Configuration for Context Grounding indexes."""
    
    def __init__(
        self,
        name: str,
        description: str,
        source_bucket: str,
        source_path: str = "/documents",
        advanced_ingestion: bool = True,
        auto_refresh: bool = True
    ):
        self.name = name
        self.description = description
        self.source_bucket = source_bucket
        self.source_path = source_path
        self.advanced_ingestion = advanced_ingestion
        self.auto_refresh = auto_refresh


class ContextGroundingService:
    """
    Service for UiPath Context Grounding integration using the UiPath SDK.
    Handles document knowledge base, search, and retrieval operations.
    """
    
    def __init__(self):
        """Initialize Context Grounding service with UiPath SDK."""
        
        # UiPath Chat Model for query enhancement
        self.chat_model = UiPathAzureChatOpenAI(
            model="gpt-4o-2024-08-06",
            temperature=0,
            max_tokens=2000,
            timeout=30,
            max_retries=2
        )
        
        # Context Grounding retrievers and vector stores
        self.document_retriever = ContextGroundingRetriever(
            index_name="LTL Claims Processing"
        )
        
        self.knowledge_retriever = ContextGroundingRetriever(
            index_name="LTL Claims Processing"
        )
        
        self.vectorstore = ContextGroundingVectorStore(
            index_name="LTL Claims Processing"
        )
        
        # Index configurations
        self.index_configs = {
            "main": IndexConfig(
                name="LTL Claims Processing",
                description="LTL Claims processing knowledge base with policies, procedures, and documents",
                source_bucket="ltl-claims-processing",
                source_path="/knowledge"
            ),
            "documents": IndexConfig(
                name="LTL_Claims_Documents",
                description="LTL Claims document repository for search and retrieval",
                source_bucket="ltl-claims-documents",
                source_path="/documents"
            ),
            "policies": IndexConfig(
                name="LTL_Claims_Policies",
                description="Claims processing policies and procedures",
                source_bucket="ltl-claims-policies",
                source_path="/policies"
            )
        }

    async def search_documents(
        self,
        query: str,
        index_name: Optional[str] = None,
        document_type: Optional[str] = None,
        max_results: int = 10,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search document knowledge base using UiPath Context Grounding.
        
        Args:
            query: Search query
            index_name: Specific index to search (defaults to documents index)
            document_type: Optional document type filter
            max_results: Maximum number of results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of relevant document excerpts with metadata and scores
        """
        try:
            index_name = index_name or "LTL Claims Processing"
            logger.info(f"🔍 Searching documents in {index_name}: {query}")
            
            # Enhance query using AI if needed
            enhanced_query = await self._enhance_search_query(query, document_type)
            
            # Use UiPath Context Grounding Retriever for search
            retriever = ContextGroundingRetriever(index_name=index_name)
            search_results = await retriever.ainvoke(enhanced_query)
            
            # Process and format results
            formatted_results = []
            if isinstance(search_results, list):
                for result in search_results:
                    # Handle Document objects from retriever
                    content = result.page_content if hasattr(result, 'page_content') else str(result)
                    metadata = result.metadata if hasattr(result, 'metadata') else {}
                    score = metadata.get('score', 1.0)  # Default score if not provided
                    
                    # Filter by minimum score
                    if score >= min_score:
                        formatted_result = {
                            "content": content,
                            "score": score,
                            "source": metadata.get('source', 'unknown'),
                            "metadata": metadata,
                            "document_type": document_type or "unknown",
                            "index_name": index_name,
                            "query_used": enhanced_query
                        }
                        formatted_results.append(formatted_result)
            
            # Sort by score descending
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"✅ Found {len(formatted_results)} relevant documents (min score: {min_score})")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return []

    async def search_knowledge_base(
        self,
        query: str,
        knowledge_type: str = "general",
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base for policies, procedures, and historical data.
        
        Args:
            query: Search query
            knowledge_type: Type of knowledge (general, policies, procedures, historical)
            max_results: Maximum number of results
            
        Returns:
            List of relevant knowledge base entries
        """
        try:
            logger.info(f"🧠 Searching knowledge base ({knowledge_type}): {query}")
            
            # Select appropriate index based on knowledge type
            index_mapping = {
                "policies": "LTL Claims Processing",
                "procedures": "LTL Claims Processing", 
                "historical": "LTL Claims Processing",
                "general": "LTL Claims Processing"
            }
            
            index_name = index_mapping.get(knowledge_type, "LTL Claims Processing")
            
            # Search using Context Grounding
            results = await self.search_documents(
                query=query,
                index_name=index_name,
                max_results=max_results,
                min_score=0.3  # Higher threshold for knowledge base
            )
            
            # Add knowledge type metadata
            for result in results:
                result["knowledge_type"] = knowledge_type
            
            logger.info(f"✅ Found {len(results)} knowledge base entries")
            return results
            
        except Exception as e:
            logger.error(f"❌ Knowledge base search failed: {e}")
            return []



    async def similarity_search_with_scores(
        self,
        query: str,
        index_name: Optional[str] = None,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Perform similarity search with relevance scores using vector store.
        
        Args:
            query: Search query
            index_name: Index name (uses default if not specified)
            k: Number of results to return
            score_threshold: Minimum score threshold
            
        Returns:
            List of (content, score) tuples
        """
        try:
            logger.info(f"🎯 Similarity search with scores: {query}")
            
            # Use vector store for similarity search
            if index_name and index_name != "LTL Claims Processing":
                # Create vector store for specific index
                vectorstore = ContextGroundingVectorStore(index_name=index_name)
            else:
                vectorstore = self.vectorstore
            
            # Perform similarity search with scores
            results = await vectorstore.asimilarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by score threshold
            filtered_results = [
                (doc.page_content, score) 
                for doc, score in results 
                if score >= score_threshold
            ]
            
            logger.info(f"✅ Similarity search complete: {len(filtered_results)} results")
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []

    async def _enhance_search_query(
        self,
        query: str,
        document_type: Optional[str] = None
    ) -> str:
        """
        Enhance search query using AI to improve search results.
        
        Args:
            query: Original search query
            document_type: Optional document type context
            
        Returns:
            Enhanced search query
        """
        try:
            # For simple queries, return as-is
            if len(query.split()) <= 3:
                return query
            
            # Use AI to enhance complex queries
            enhancement_prompt = f"""
            Enhance this search query for better document retrieval in an LTL claims system:
            
            Original query: "{query}"
            Document type: {document_type or "any"}
            
            Provide a more specific, keyword-rich query that would find relevant documents.
            Focus on key terms related to LTL shipping, claims, damage, carriers, etc.
            
            Enhanced query:
            """
            
            response = await self.chat_model.ainvoke([
                {"role": "system", "content": "You are an expert at creating search queries for LTL claims documents."},
                {"role": "user", "content": enhancement_prompt}
            ])
            
            enhanced_query = response.content.strip()
            
            # Fallback to original if enhancement fails
            if not enhanced_query or len(enhanced_query) > 200:
                return query
            
            logger.debug(f"🔍 Query enhanced: '{query}' -> '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.warning(f"⚠️ Query enhancement failed: {e}")
            return query




# Global context grounding service instance
context_grounding_service = ContextGroundingService()