# Context Grounding Retriever

The `ContextGroundingRetriever` is a document retrieval system that uses vector search to efficiently find and retrieve relevant information from your document store.

## Overview

`ContextGroundingRetriever` allows you to:
- Search through indexed documents using natural language queries
- Ground LLM responses in your organization's specific information
- Retrieve context-relevant documents for various applications


## Basic Usage

Create a simple retriever by specifying an index name:

```python
from uipath_langchain.retrievers import ContextGroundingRetriever

retriever = ContextGroundingRetriever(index_name = "Company Policy Context")
pprint(retriever.invoke("What is the company policy on remote work?"))
```

## Integration with LangChain Tools

You can easily integrate the retriever with LangChain's tool system:

```python
from langchain.tools.retriever import create_retriever_tool
from uipath_langchain.retrievers import ContextGroundingRetriever

retriever = ContextGroundingRetriever(index_name = "Company Policy Context")
retriever_tool = create_retriever_tool(
    retriever,
    "ContextforInvoiceDisputeInvestigation",
   """
   Use this tool to search the company internal documents for information about policies around dispute resolution.
   Use a meaningful query to load relevant information from the documents. Save the citation for later use.
   """
)
```


## Advanced Usage

For complex applications, the retriever can be combined with other LangChain components to create robust document QA systems, agents, or knowledge bases.