"""UiPath LLM wrapper for chat completions."""

from __future__ import annotations

from typing import List, Tuple, Union, Optional

from uipath import UiPath


def call_llm(
    uipath: UiPath,
    messages: Union[List[Dict[str, str]], List[Tuple[str, str]]],
    model: str = "gpt-4o-mini-2024-07-18",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Call UiPath LLM Gateway for chat completions.

    Parameters
    ----------
    uipath : UiPath
        UiPath SDK instance
    messages : Union[List[Dict[str, str]], List[Tuple[str, str]]]
        Messages in format [("system", "content"), ("user", "content")]
        or [{"role": "system", "content": "..."}, ...]
    model : str
        Model identifier
    temperature : float
        Temperature for generation (0.0-1.0)
    max_tokens : int
        Maximum tokens to generate

    Returns
    -------
    str
        Generated text response

    Raises
    ------
    Exception
        If LLM call fails
    """
    # Call UiPath LLM Gateway
    response = uipath.llm.chat_completions(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )

    # Extract text from response
    if hasattr(response, 'choices') and len(response.choices) > 0:
        return response.choices[0].message.content.strip()

    raise ValueError("Invalid response from LLM")


def call_llm_with_fallback(
    uipath: Optional[UiPath],
    messages: Union[List[Dict[str, str]], List[Tuple[str, str]]],
    model: str = "gpt-4o-mini-2024-07-18",
    temperature: float = 0.0,
    max_tokens: int = 512,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
) -> str:
    """
    Call LLM with fallback to direct OpenAI-compatible endpoint.

    Tries UiPath first, falls back to direct API call if UiPath unavailable.

    Parameters
    ----------
    uipath : Optional[UiPath]
        UiPath SDK instance (if available)
    messages : Union[List[Dict[str, str]], List[Tuple[str, str]]]
        Messages for chat
    model : str
        Model identifier
    temperature : float
        Temperature for generation
    max_tokens : int
        Maximum tokens to generate
    openai_api_key : Optional[str]
        API key for fallback (if not using UiPath)
    openai_base_url : Optional[str]
        Base URL for fallback (e.g., OpenRouter)

    Returns
    -------
    str
        Generated text response
    """
    # Note: UiPath LLM Gateway is async-only in Python SDK
    # For sync contexts, use direct LangChain integration

    # Fallback: Use LangChain with direct API
    if openai_api_key and openai_base_url:
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Convert messages to prompt
        if isinstance(messages[0], tuple):
            # Format: [("system", "content"), ("user", "content")]
            prompt_template = "\n\n".join([f"{role}: {content}" for role, content in messages])
        else:
            # Format: [{"role": "system", "content": "..."}, ...]
            prompt_template = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()

        return chain.invoke({})

    raise ValueError("No LLM backend available (UiPath or fallback credentials missing)")
