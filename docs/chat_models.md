# Chat Models

UiPath provides chat models that are compatible with LangGraph as drop-in replacements for LLM clients. You do not need to add tokens from OpenAI, Vertex AI or Anthropic, usage of these chat models will consume `Agent Units` on your account.

## UiPathChatOpenAI and UiPathAzureChatOpenAI

`UiPathChatOpenAI` and `UiPathAzureChatOpenAI` are both compatible as drop-in replacements for `ChatOpenAI` and `AzureChatOpenAI` from `langchain_openai`.

> **Note:** `UiPathAzureChatOpenAI` will be deprecated in the future. We recommend using `UiPathChatOpenAI` for all new implementations.

### Example usage

Here is code using `ChatOpenAI` from `langchain_openai`:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=4000,
    timeout=30,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instead of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)
```

You can simply change `ChatOpenAI` with `UiPathChatOpenAI`, you don't have to provide an OpenAI API key.

```python
from uipath_langchain.chat import UiPathChatOpenAI

llm = UiPathChatOpenAI(
    model="gpt-4o-2024-08-06",
    temperature=0,
    max_tokens=4000,
    timeout=30,
    max_retries=2,
    # other params...
)
```

Currently, the following models can be used with `UiPathChatOpenAI` (this list can be updated in the future):

-   `gpt-4o-2024-05-13`, `gpt-4o-2024-08-06`, `gpt-4o-2024-11-20`, `gpt-4o-mini-2024-07-18`, `gpt-4.1-2025-04-14`, `gpt-4.1-mini-2025-04-14`, `gpt-4.1-nano-2025-04-14`, `gpt-5-2025-08-07`, `gpt-5-chat-2025-08-07`, `gpt-5-mini-2025-08-07`, `gpt-5-nano-2025-08-07`, `gpt-5.1-2025-11-13`, `gpt-5.2-2025-12-11`

## UiPathChatBedrock and UiPathChatBedrockConverse

`UiPathChatBedrock` and `UiPathChatBedrockConverse` can be used as drop in replacements for `ChatBedrock` and `ChatBedrockConverse` from `langchain_aws`.

### Installation

These classes require additional dependencies. Install them with:

```bash
pip install uipath-langchain[bedrock]
# or using uv:
uv add 'uipath-langchain[bedrock]'
```

### Example usage

Here is code using `ChatBedrockConverse` from `langchain_aws`:

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=1024,
)
```

You can replace it with `UiPathChatBedrockConverse` without needing AWS credentials:

```python
from uipath_langchain.chat import UiPathChatBedrockConverse

llm = UiPathChatBedrockConverse(
    model_name="anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature=0,
    max_tokens=1024,
    # other params...
)
```

Similarly, `ChatBedrock` can be replaced with `UiPathChatBedrock`:

```python
from uipath_langchain.chat import UiPathChatBedrock

llm = UiPathChatBedrock(
    model_name="anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature=0,
    max_tokens=1024,
    # other params...
)
```

Currently, the following models can be used (this list can be updated in the future):

-   `anthropic.claude-3-7-sonnet-20250219-v1:0`, `anthropic.claude-sonnet-4-20250514-v1:0`, `anthropic.claude-sonnet-4-5-20250929-v1:0`, `anthropic.claude-haiku-4-5-20251001-v1:0`

## UiPathChatVertex

`UiPathChatVertex` can be used as a drop in replacement for `ChatGoogleGenerativeAI` from `langchain_google_genai`.

### Installation

This class requires additional dependencies. Install them with:

```bash
pip install uipath-langchain[vertex]
# or using uv:
uv add 'uipath-langchain[vertex]'
```

### Example usage

Here is code using `ChatGoogleGenerativeAI` from `langchain_google_genai`:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=1024,
    # api_key="...",
    # other params...
)
```

You can replace it with `UiPathChatVertex` without needing a Google API key:

```python
from uipath_langchain.chat import UiPathChatVertex

llm = UiPathChatVertex(
    model_name="gemini-2.5-flash",
    temperature=0,
    max_tokens=1024,
    # other params...
)
```

Currently, the following models can be used (this list can be updated in the future):

-   `gemini-2.0-flash-001`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-pro-preview`

## UiPathChat

`UiPathChat` is a more versatile class that can suport models from diferent vendors including OpenAI.

### Example usage

Given the following code:

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    # other params...
)
```

You can replace it with `UiPathChat` like so:

```python
from uipath_langchain.chat.models import UiPathChat

llm = UiPathChat(
    model="anthropic.claude-3-opus-20240229-v1:0",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    # other params...
)
```

Currently the following models can be used with `UiPathChat` (this list can be updated in the future):

-   `anthropic.claude-3-5-sonnet-20240620-v1:0`, `anthropic.claude-3-5-sonnet-20241022-v2:0`, `anthropic.claude-3-7-sonnet-20250219-v1:0`, `anthropic.claude-3-haiku-20240307-v1:0`, `gemini-1.5-pro-001`, `gemini-2.0-flash-001`, `gpt-4o-2024-05-13`, `gpt-4o-2024-08-06`, `gpt-4o-2024-11-20`, `gpt-4o-mini-2024-07-18`, `o3-mini-2025-01-31`

/// warning
Please note that you may get errors related to data residency, as some models are not available on all regions.

Example: `[Enforced Region] No model configuration found for product uipath-python-sdk in EU using model anthropic.claude-3-opus-20240229-v1:0`.

///
