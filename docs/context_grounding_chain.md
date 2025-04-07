# ContextGroundingVectorStore

`ContextGroundingVectorStore` is a vector store implementation designed for context-aware document retrieval. It allows you to perform semantic searches and create retrieval chains with language models.

## Initialization

You can initialize the vector store with an index name:

```python
    from uipath_langchain.vectorstores.context_grounding_vectorstore import ContextGroundingVectorStore

    vectorstore = ContextGroundingVectorStore(index_name=index_name)
```

## Searching Documents

The vector store supports various search methods:

```python
    # Perform semantic searches with distance scores
    docs_with_scores = await vectorstore.asimilarity_search_with_score(query=query, k=5)

    # Perform a similarity search with relevance scores
    docs_with_relevance_scores = await vectorstore.asimilarity_search_with_relevance_scores(query=query, k=5)
```

## Creating a Retrieval Chain

You can integrate the vector store into a retrieval chain with a language model:

```python
    # Run a retrieval chain
    model = UiPathAzureChatOpenAI(model="gpt-4o-2024-08-06", max_retries=3)
    retrieval_chain = create_retrieval_chain(vectorstore=vectorstore, model=model)

    query = "What is the ECCN for a laptop?"
    result = retrieval_chain(query)
```