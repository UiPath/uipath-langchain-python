def get_preload_modules() -> list[str]:
    return [
        "mcp",
        "mcp.client.streamable_http",
        "langchain_mcp_adapters.tools",
        "uipath_langchain.chat.models",
        "uipath_langchain.chat.openai",
        "uipath_langchain.chat.bedrock",
        "uipath_langchain.chat.vertex",
        "uipath_langchain.chat.supported_models",
        "uipath_langchain.chat.types",
    ]
