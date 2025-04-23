# ContextGroundingRetriever

The `ContextGroundingRetriever` is a document retrieval system that uses vector search to efficiently find and retrieve relevant information from your document store.

## Overview

`ContextGroundingRetriever` allows you to:
- Search through indexed documents using natural language queries
- Ground LLM responses in your organization's specific information
- Retrieve context-relevant documents for various applications


You will need to create an index in `Context Grounding` to use this feature. To create an index go to organization `Admin` -> `AI Trust Layer` -> `Context Grounding`. There you can create a new index and add documents to it. See the full documentation [here](https://docs.uipath.com/automation-cloud/automation-cloud/latest/admin-guide/about-context-grounding) for more details.

## Basic Usage

Create a simple retriever by specifying an index name:

```python
from uipath_langchain.retrievers import ContextGroundingRetriever

retriever = ContextGroundingRetriever(index_name = "Company Policy Context")
print(retriever.invoke("What is the company policy on remote work?"))
```

## Integration with LangChain Tools

You can easily integrate the retriever with LangChain's tool system:

```python
from langchain.agents import create_react_agent
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

# You can use the tool in your agents
model = OpenAI()
tools = [retriever_tool]
agent = create_react_agent(model, tools, prompt="Answer user questions as best as you can using the search tool.")
```


## Advanced Usage

For complex applications, the retriever can be combined with other LangChain components to create robust document QA systems, agents, or knowledge bases.