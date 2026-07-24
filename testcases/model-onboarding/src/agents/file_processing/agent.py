"""File-processing coded agent.

Coded (LangGraph-compatible) equivalent of the low-code ``FileProcessingAgent``
authored in Studio Web / Agent Builder. The low-code version accepts a
``job-attachment`` input plus a ``prompt`` and uses the built-in *Analyze Files*
tool to read PDF/image contents and answer the task.

There is no automated low-code -> coded eject in the tooling, so this is a
faithful re-implementation: the same system prompt and the same single-file
analysis behavior, expressed against ``uipath_langchain``'s multimodal invoke
helper. The model (and thus API flavor) is supplied by the caller, which builds
it from ``model_spec`` in ``input.json`` — so this agent runs against whatever
model/path the onboarding matrix is exercising.

Mirrors the low-code ``agent.json``:
- system prompt: file-processing assistant that reads the file then answers
- input: a task ``prompt`` + one file
- output: a text analysis grounded in the file
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from uipath_langchain.agent.multimodal.invoke import llm_call_with_files
from uipath_langchain.agent.multimodal.types import FileInfo

NAME = "file_processing"

# Kept in sync with the low-code agent's system message (agent.json).
SYSTEM_PROMPT = (
    "You are a file-processing assistant. You are given a single file "
    "(PDF or image) and a task. Read the file's contents, then answer the task "
    "concisely based only on what the file contains. If the file cannot be "
    "read, say so plainly."
)


async def run(model: BaseChatModel, prompt: str, files: list[FileInfo]) -> str:
    """Run the file-processing agent over one file.

    Args:
        model: The chat model to use (already built for the target path/flavor).
        prompt: The task/question to answer about the file.
        files: Exactly the files to attach for this cell (one per invocation in
            the onboarding grid). An empty list degrades to a plain text call.

    Returns:
        ``"✓"`` on a non-empty grounded answer, otherwise ``"✗ ..."``.
    """
    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response: AIMessage = await llm_call_with_files(messages, files, model)
    if response.content and str(response.content).strip():
        return "✓"
    return "✗ empty response"
