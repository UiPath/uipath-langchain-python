# Chat Models

UiPath provides chat model classes compatible with LangChain and LangGraph as drop-in replacements. You do not need provider API keys — usage consumes Agent Units on your account.

To see the models available for your account, run:

```bash
uipath list-models
```

/// note
`UiPathChat` and `UiPathAzureChatOpenAI` are legacy classes and will be phased out in a future release. `UiPathAzureChatOpenAI` has been renamed to `UiPathChatOpenAI` (same class, new name). When you can, migrate existing code to one of the provider-specific classes below.
///

## UiPathChatOpenAI

Drop-in replacement for `ChatOpenAI` or `AzureChatOpenAI`. This is the new name for `UiPathAzureChatOpenAI`.

Original code using `ChatOpenAI`:

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

Swap `ChatOpenAI` for `UiPathChatOpenAI` — no OpenAI token needed:

```python
from uipath_langchain.chat import UiPathChatOpenAI

llm = UiPathChatOpenAI(
    model="<model>",
    temperature=0,
    max_tokens=4000,
    timeout=30,
    max_retries=2,
    # other params...
)
```

## UiPathChatAnthropicBedrock

Drop-in replacement for `ChatAnthropic` (or `ChatBedrock` when using Anthropic models).

Original code using `ChatAnthropic`:

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

Swap for `UiPathChatAnthropicBedrock`:

```python
from uipath_langchain.chat import UiPathChatAnthropicBedrock

llm = UiPathChatAnthropicBedrock(
    model_id="<model>",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    # other params...
)
```

## UiPathChatBedrockConverse

Drop-in replacement for `ChatBedrockConverse` from `langchain_aws`. Supports models from multiple providers via the AWS Bedrock Converse API.

Original code using `ChatBedrockConverse`:

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=1024,
    # other params...
)
```

Swap for `UiPathChatBedrockConverse`:

```python
from uipath_langchain.chat import UiPathChatBedrockConverse

llm = UiPathChatBedrockConverse(
    model="<model>",
    temperature=0,
    max_tokens=1024,
    # other params...
)
```

## UiPathChatVertex

Drop-in replacement for `ChatGoogleGenerativeAI` or `ChatVertexAI`.

Original code using `ChatGoogleGenerativeAI`:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=1024,
    # other params...
)
```

Swap for `UiPathChatVertex`:

```python
from uipath_langchain.chat import UiPathChatVertex

llm = UiPathChatVertex(
    model="<model>",
    temperature=0,
    max_tokens=1024,
    # other params...
)
```

/// warning
Some models may not be available in all regions due to data residency restrictions. Use `uipath list-models` to see which models are available in your region.
///
