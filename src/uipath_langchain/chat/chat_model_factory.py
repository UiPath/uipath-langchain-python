from typing import Any

from langchain_core.language_models import BaseChatModel

from uipath_langchain.chat.types import APIFlavor, LLMProvider

_DEFAULT_API_FLAVOR: dict[LLMProvider, APIFlavor] = {
    LLMProvider.OPENAI: APIFlavor.OPENAI_RESPONSES,
    LLMProvider.BEDROCK: APIFlavor.AWS_BEDROCK_CONVERSE,
    LLMProvider.VERTEX: APIFlavor.VERTEX_GEMINI_GENERATE_CONTENT,
}

_API_FLAVOR_TO_PROVIDER: dict[APIFlavor, LLMProvider] = {
    APIFlavor.OPENAI_RESPONSES: LLMProvider.OPENAI,
    APIFlavor.OPENAI_COMPLETIONS: LLMProvider.OPENAI,
    APIFlavor.AWS_BEDROCK_CONVERSE: LLMProvider.BEDROCK,
    APIFlavor.AWS_BEDROCK_INVOKE: LLMProvider.BEDROCK,
    APIFlavor.VERTEX_GEMINI_GENERATE_CONTENT: LLMProvider.VERTEX,
    APIFlavor.VERTEX_ANTHROPIC_CLAUDE: LLMProvider.VERTEX,
}


def _fetch_discovery(agenthub_config: str) -> list[dict[str, Any]]:
    """Fetch available models from LLM Gateway discovery endpoint."""
    from uipath.platform import UiPath

    sdk = UiPath()
    models = sdk.agenthub.get_available_llm_models(
        headers={"X-UiPath-AgentHub-Config": agenthub_config}
    )
    return [model.model_dump(by_alias=True) for model in models]


def _create_openai_llm(
    model: str,
    api_flavor: APIFlavor,
    temperature: float,
    max_tokens: int,
    agenthub_config: str,
    byo_connection_id: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create UiPathChatOpenAI for OpenAI models via LLMGateway."""
    from uipath_langchain.chat.openai import UiPathChatOpenAI

    azure_open_ai_latest_api_version = "2025-04-01-preview"

    match api_flavor:
        case APIFlavor.OPENAI_RESPONSES:
            return UiPathChatOpenAI(
                use_responses_api=True,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_version=azure_open_ai_latest_api_version,
                agenthub_config=agenthub_config,
                byo_connection_id=byo_connection_id,
                output_version="v1",
                **kwargs,
            )
        case APIFlavor.OPENAI_COMPLETIONS:
            return UiPathChatOpenAI(
                use_responses_api=False,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_version=azure_open_ai_latest_api_version,
                agenthub_config=agenthub_config,
                byo_connection_id=byo_connection_id,
                output_version="v1",
                **kwargs,
            )
        case _:
            raise ValueError(f"Unknown api_flavor={api_flavor} for OpenAI")


def _create_bedrock_llm(
    model: str,
    api_flavor: APIFlavor,
    temperature: float,
    max_tokens: int,
    agenthub_config: str,
    byo_connection_id: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create UiPathChatBedrockConverse for Claude models via LLMGateway."""
    from uipath_langchain.chat.bedrock import (
        UiPathChatBedrock,
        UiPathChatBedrockConverse,
    )

    match api_flavor:
        case APIFlavor.AWS_BEDROCK_CONVERSE:
            return UiPathChatBedrockConverse(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                agenthub_config=agenthub_config,
                byo_connection_id=byo_connection_id,
                output_version="v1",
                **kwargs,
            )
        case APIFlavor.AWS_BEDROCK_INVOKE:
            return UiPathChatBedrock(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                agenthub_config=agenthub_config,
                byo_connection_id=byo_connection_id,
                output_version="v1",
                **kwargs,
            )
        case _:
            raise ValueError(f"Unknown api_flavor={api_flavor} for AwsBedrock")


def _create_vertex_llm(
    model: str,
    api_flavor: APIFlavor,
    temperature: float,
    max_tokens: int | None,
    agenthub_config: str,
    byo_connection_id: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create UiPathChatVertex for Gemini models via LLMGateway."""
    from uipath_langchain.chat.vertex import UiPathChatVertex

    match api_flavor:
        case APIFlavor.VERTEX_GEMINI_GENERATE_CONTENT:
            return UiPathChatVertex(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                agenthub_config=agenthub_config,
                byo_connection_id=byo_connection_id,
                output_version="v1",
                **kwargs,
            )
        case APIFlavor.VERTEX_ANTHROPIC_CLAUDE:
            raise ValueError(f"api_flavor={api_flavor} is not yet supported for Vertex")
        case _:
            raise ValueError(f"Unknown api_flavor={api_flavor} for Vertex")


def _resolve_vendor(api_flavor: APIFlavor) -> LLMProvider:
    return _API_FLAVOR_TO_PROVIDER[api_flavor]


def _resolve_api_flavor(vendor: LLMProvider, model_name: str) -> APIFlavor:
    if vendor == LLMProvider.VERTEX and "claude" in model_name:
        return APIFlavor.VERTEX_ANTHROPIC_CLAUDE
    return _DEFAULT_API_FLAVOR[vendor]


def _compute_vendor_and_api_flavor(
    model: dict[str, Any],
) -> tuple[LLMProvider, APIFlavor]:
    vendor = model.get("vendor")
    api_flavor = model.get("apiFlavor")
    model_name = model.get("modelName", "")

    if api_flavor is None and vendor is None:
        raise ValueError(
            f"Neither vendor nor apiFlavor provided for model '{model_name}'. "
            "At least one must be present."
        )

    if api_flavor is not None and api_flavor not in [p.value for p in APIFlavor]:
        raise ValueError(
            f"Unknown apiFlavor '{api_flavor}' for model '{model_name}'. "
            f"Supported apiFlavors: {[p.value for p in APIFlavor]}"
        )

    if vendor is not None and vendor not in [p.value for p in LLMProvider]:
        raise ValueError(
            f"Unknown vendor '{vendor}' for model '{model_name}'. "
            f"Supported vendors: {[p.value for p in LLMProvider]}"
        )

    resolved_vendor: LLMProvider
    resolved_api_flavor: APIFlavor

    if vendor is None and api_flavor is not None:
        resolved_api_flavor = APIFlavor(api_flavor)
        resolved_vendor = _resolve_vendor(resolved_api_flavor)
    elif api_flavor is None and vendor is not None:
        resolved_vendor = LLMProvider(vendor)
        resolved_api_flavor = _resolve_api_flavor(resolved_vendor, model_name)
    else:
        assert vendor is not None and api_flavor is not None
        resolved_vendor = LLMProvider(vendor)
        resolved_api_flavor = APIFlavor(api_flavor)

    return resolved_vendor, resolved_api_flavor


def _get_model_info(
    model: str,
    agenthub_config: str,
    byo_connection_id: str | None,
) -> dict[str, Any]:
    discovery_models = _fetch_discovery(agenthub_config)

    matching_models = [m for m in discovery_models if m.get("modelName") == model]

    if byo_connection_id:
        matching_models = [
            m
            for m in matching_models
            if (byom_details := m.get("byomDetails"))
            and byom_details.get("integrationServiceConnectionId", "").lower()
            == byo_connection_id.lower()
        ]

    if not byo_connection_id and len(matching_models) > 1:
        matching_models = [m for m in matching_models if m.get("byomDetails") is None]

    if not matching_models:
        raise ValueError(
            f"model='{model}' and byo_connection_id={byo_connection_id}"
            + " is not available. It was not returned by the discovery API."
        )

    return matching_models[0]


def get_chat_model(
    model: str,
    temperature: float,
    max_tokens: int,
    agenthub_config: str,
    byo_connection_id: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create and configure LLM instance using LLMGateway API.

    Fetches available models from the discovery API and selects the appropriate
    LLM class based on the apiFlavor field from the matching model configuration.
    """
    model_info = _get_model_info(model, agenthub_config, byo_connection_id)

    vendor, api_flavor = _compute_vendor_and_api_flavor(model_info)
    model_name: str = model_info.get("modelName", model)

    match LLMProvider(vendor):
        case LLMProvider.OPENAI:
            return _create_openai_llm(
                model_name,
                api_flavor,
                temperature,
                max_tokens,
                agenthub_config,
                byo_connection_id,
                **kwargs,
            )
        case LLMProvider.BEDROCK:
            return _create_bedrock_llm(
                model_name,
                api_flavor,
                temperature,
                max_tokens,
                agenthub_config,
                byo_connection_id,
                **kwargs,
            )
        case LLMProvider.VERTEX:
            return _create_vertex_llm(
                model_name,
                api_flavor,
                temperature,
                max_tokens,
                agenthub_config,
                byo_connection_id,
                **kwargs,
            )
