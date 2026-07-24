"""IS-tools coded agent.

Tests that the model under test can drive an **Integration Service activity
tool** end to end, across the BYO LLM vendor connector flavors available in
the tenant. For each selected flavor the agent:

1. binds one LangChain tool (``ask_llm_via_gateway``) whose executor invokes
   that flavor's IS activity via ``sdk.connections.invoke_activity_async``,
2. sends a prompt that forces the model to call the tool,
3. executes the activity through the flavor's IS connection,
4. feeds the ``ToolMessage`` back and requires a non-empty final answer.

The flavor registry below was built from live ``uip is resources describe``
output against the ``llm_gateway_automated_testing`` alpha tenant (paths,
methods, query/path/multipart routing, and required fields are all verified,
not guessed). Connection ids and vendor model names are **defaults for that
tenant** — override both per flavor via ``model_spec.is_tools`` in
``input.json`` when running against another org.

Flavors (6 — what the IS connectors actually expose; e.g. Bedrock's connector
has only a converse activity, so there is no separate invoke flavor here):

======================  =======================================  ============
flavor                  connector / activity                     body style
======================  =======================================  ============
``azure_openai``        microsoft-azureopenai / generateChat…    JSON + query
``openai``              openai-openai / v2::chat::completion     multipart
``openai_v1``           openai-openaiv1compliant / chatCompl…    JSON
``bedrock_converse``    aws-bedrock / completion::converse       multipart+qry
``vertex``              google-vertex / textCompletionUsingG…    multipart+path
``anthropic``           anthropic-claude / messages              JSON
======================  =======================================  ============
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

NAME = "is_tools"

# Prompt that forces the model under test to call the IS tool.
IS_TOOL_PROMPT = (
    "Use the ask_llm_via_gateway tool to ask this exact question: "
    "'Reply with the single word OK.' Then report what the tool returned."
)


@dataclass
class IsFlavor:
    """One IS activity flavor: how to build and invoke the activity."""

    connector_key: str
    object_path: str
    default_connection_id: str
    default_model: str
    # Builds (activity_metadata_kwargs, activity_input) for a question.
    build: Callable[[str, str], tuple[dict[str, Any], dict[str, Any]]] = field(
        repr=False, default=None  # type: ignore[assignment]
    )


def _azure_openai(model: str, question: str) -> tuple[dict, dict]:
    meta = {
        "object_path": "/generateChatCompletionConsolidated",
        "method_name": "POST",
        "content_type": "application/json",
        "query_params": ["modelId", "api-version"],
        "body_fields": ["prompt", "knowledge_base", "max_tokens", "temperature"],
    }
    body = {
        "modelId": model,
        "prompt": question,
        "knowledge_base": False,
        "max_tokens": 100,
        "temperature": 0,
    }
    return meta, body


def _openai(model: str, question: str) -> tuple[dict, dict]:
    meta = {
        "object_path": "/v2/chat/completion",
        "method_name": "POST",
        "content_type": "multipart/form-data",
        "multipart_params": ["body"],
        "json_body_section": "body",
    }
    body = {
        "body": {
            "model": model,
            "prompt": question,
            "max_tokens": 100,
            "temperature": 0,
        }
    }
    return meta, body


def _openai_v1(model: str, question: str) -> tuple[dict, dict]:
    meta = {
        "object_path": "/chatCompletion",
        "method_name": "POST",
        "content_type": "application/json",
        "body_fields": ["model", "prompt"],
    }
    return meta, {"model": model, "prompt": question}


def _bedrock_converse(model: str, question: str) -> tuple[dict, dict]:
    meta = {
        "object_path": "/completion/converse",
        "method_name": "POST",
        "content_type": "multipart/form-data",
        "query_params": ["modelName"],
        "multipart_params": ["body"],
        "json_body_section": "body",
    }
    body = {"modelName": model, "body": {"prompt": question, "maxTokens": 100}}
    return meta, body


def _vertex(model: str, question: str) -> tuple[dict, dict]:
    meta = {
        # modelName is a PATH parameter on this activity.
        "object_path": f"/textCompletionUsingGemini/{model}",
        "method_name": "POST",
        "content_type": "multipart/form-data",
        "multipart_params": ["body"],
        "json_body_section": "body",
    }
    return meta, {"body": {"prompt": question, "maxOutputTokens": 100}}


def _anthropic(model: str, question: str) -> tuple[dict, dict]:
    meta = {
        "object_path": "/messages",
        "method_name": "POST",
        "content_type": "application/json",
        "body_fields": ["model", "prompt", "maxTokens"],
    }
    return meta, {"model": model, "prompt": question, "maxTokens": 100}


# Connection ids default to the llm_gateway_automated_testing alpha tenant
# (all verified Enabled via `uip is connections ping`). Override per org via
# model_spec.is_tools.connections.
#
# default_model caveat: azure_openai and openai_v1 (Azure/Foundry-backed)
# expect a DEPLOYMENT name specific to the connection — a wrong one fails with
# DeploymentNotFound (observed live). anthropic/bedrock/vertex model ids are
# vendor-global; the anthropic default was verified live end to end.
FLAVOR_REGISTRY: dict[str, IsFlavor] = {
    "azure_openai": IsFlavor(
        connector_key="uipath-microsoft-azureopenai",
        object_path="/generateChatCompletionConsolidated",
        default_connection_id="fda2fdd1-a0ca-4a8a-bbcc-01e2a908d2ce",
        default_model="gpt-4o-mini",
        build=_azure_openai,
    ),
    "openai": IsFlavor(
        connector_key="uipath-openai-openai",
        object_path="/v2/chat/completion",
        default_connection_id="5bc09bd6-adfa-47da-b7a6-13725b9a0404",
        default_model="gpt-4o-mini",
        build=_openai,
    ),
    "openai_v1": IsFlavor(
        connector_key="uipath-openai-openaiv1compliant",
        object_path="/chatCompletion",
        default_connection_id="2c4118a4-f27d-4354-a4b7-bcc6e2ecaf07",
        default_model="gpt-4o-mini",
        build=_openai_v1,
    ),
    "bedrock_converse": IsFlavor(
        connector_key="uipath-aws-bedrock",
        object_path="/completion/converse",
        default_connection_id="6452e8dc-64df-4e48-83c5-bc6e17945340",
        default_model="anthropic.claude-haiku-4-5-20251001-v1:0",
        build=_bedrock_converse,
    ),
    "vertex": IsFlavor(
        connector_key="uipath-google-vertex",
        object_path="/textCompletionUsingGemini/{modelName}",
        default_connection_id="3cbbb133-3557-409d-8c75-7528121d6f0a",
        default_model="gemini-2.5-flash",
        build=_vertex,
    ),
    "anthropic": IsFlavor(
        connector_key="uipath-anthropic-claude",
        object_path="/messages",
        default_connection_id="b631c4bb-8719-4cb7-823a-7aef6dab9766",
        default_model="claude-haiku-4-5-20251001",
        build=_anthropic,
    ),
}


async def _invoke_is_activity(
    flavor: str, connection_id: str, is_model: str, question: str
) -> str:
    """Invoke one flavor's IS activity and return a compact result string.

    Isolated so tests can monkeypatch it; builds the SDK client lazily
    (module-level construction breaks scaffold/introspection tooling).
    """
    from uipath.platform import UiPath
    from uipath.platform.connections import (
        ActivityMetadata,
        ActivityParameterLocationInfo,
    )

    cfg = FLAVOR_REGISTRY[flavor]
    meta_kwargs, activity_input = cfg.build(is_model, question)

    location = ActivityParameterLocationInfo(
        query_params=meta_kwargs.pop("query_params", []),
        path_params=meta_kwargs.pop("path_params", []),
        body_fields=meta_kwargs.pop("body_fields", []),
        multipart_params=meta_kwargs.pop("multipart_params", []),
    )
    json_body_section = meta_kwargs.pop("json_body_section", None)
    metadata = ActivityMetadata(
        parameter_location_info=location,
        json_body_section=json_body_section,
        **meta_kwargs,
    )

    sdk = UiPath()
    response = await sdk.connections.invoke_activity_async(
        activity_metadata=metadata,
        connection_id=connection_id,
        activity_input=activity_input,
    )
    # Response shapes differ per vendor; a compact JSON dump is enough for the
    # model to report on, and avoids per-flavor parsing.
    return json.dumps(response, default=str)[:400]


async def run_flavor(
    model: BaseChatModel,
    flavor: str,
    connection_id: str | None = None,
    is_model: str | None = None,
) -> str:
    """Full model-driven round trip for one IS flavor.

    The model under test must request the tool; the tool executes the real IS
    activity; the result is fed back; the final answer must be non-empty.

    Returns:
        ``"✓"`` or ``"✗ ..."`` per the testcase cell contract.
    """
    if flavor not in FLAVOR_REGISTRY:
        return f"✗ unknown flavor '{flavor}'"
    cfg = FLAVOR_REGISTRY[flavor]
    conn_id = connection_id or cfg.default_connection_id
    vendor_model = is_model or cfg.default_model

    captured: dict[str, str] = {}

    @tool
    async def ask_llm_via_gateway(question: str) -> str:
        """Ask a question to an LLM through the UiPath Integration Service
        gateway and return its raw response.

        Args:
            question: The question to send to the gateway LLM.
        """
        result = await _invoke_is_activity(flavor, conn_id, vendor_model, question)
        captured["is_response"] = result
        return result

    llm = model.bind_tools([ask_llm_via_gateway])
    messages: list = [HumanMessage(content=IS_TOOL_PROMPT)]

    first = await llm.ainvoke(messages)
    if not isinstance(first, AIMessage) or not first.tool_calls:
        return "✗ no tool call requested"

    messages.append(first)
    for call in first.tool_calls:
        if call["name"] != "ask_llm_via_gateway":
            return f"✗ unexpected tool '{call['name']}'"
        try:
            result = await ask_llm_via_gateway.ainvoke(call["args"])
        except Exception as e:  # IS/vendor failures must surface per cell
            return f"✗ IS activity failed: {str(e)[:80]}"
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    if "is_response" not in captured:
        return "✗ tool executed but produced no IS response"

    final = await llm.ainvoke(messages)
    if not isinstance(final, AIMessage) or not str(final.content).strip():
        return "✗ empty final answer"
    return "✓"
