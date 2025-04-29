# Quickstart Guide: UiPath LangChain Agents

## Introduction

This guide walks you through the process of setting up, creating, publishing, and running your first UiPath-LangChain Agent.

## Prerequisites

Ensure you have the following installed before proceeding:

-   Python 3.10 or higher
-   `pip` or `uv` package manager
-   A UiPath Cloud Platform account with appropriate permissions
-   Either Anthropic or OpenAI API key

/// info

1. **Anthropic** - Generate an Anthropic API key [here](https://console.anthropic.com/settings/keys).

2. **OpenAi** - Generate an OpenAI API key [here](https://platform.openai.com).

///

## Step-by-Step Guide

## Creating a New Project

We recommend using `uv` for package management. To create a new project:

//// tab | Linux, macOS, Windows Bash

<!-- termynal -->

```shell
> mkdir example
> cd example
```

////

//// tab | Windows PowerShell

<!-- termynal -->

```powershell
> New-Item -ItemType Directory -Path example
> Set-Location example
```

////

//// tab | uv
    new: true

<!-- termynal -->

```shell
# Initialize a new uv project in the current directory
> uv init . --python 3.10

# Create a new virtual environment
# By default, uv creates a virtual environment in a directory called .venv
> uv venv
Using CPython 3.10.16 interpreter at: [PATH]
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate

# Activate the virtual environment
# For Windows PowerShell: .venv\Scripts\Activate.ps1
# For Windows Bash: source .venv/Scripts/activate
> source .venv/bin/activate

# Install the uipath package
> uv add uipath-langchain

# Verify the uipath installation
> uipath --lv
uipath-langchain version 0.0.100
```

////

//// tab | pip

<!-- termynal -->

```shell
# Create a new virtual environment
> python -m venv .venv

# Activate the virtual environment
# For Windows PowerShell: .venv\Scripts\Activate.ps1
# For Windows Bash: source .venv/Scripts/activate
> source .venv/bin/activate

# Upgrade pip to the latest version
> python -m pip install --upgrade pip

# Install the uipath package
> pip install uipath-langchain

# Verify the uipath installation
> uipath --lv
uipath-langchain version 0.0.100
```

////

## Create Your First UiPath Agent

Generate your first UiPath LangChain:

<!-- termynal -->

```shell
> uipath new my-agent
⠋ Creating new agent my-agent in current directory ...
✓  Created 'main.py' file.
✓  Created 'langgraph.json' file.
✓  Created 'pyproject.toml' file.
Resolved 90 packages in 351ms
Prepared 2 packages in 213ms
Installed 2 packages in 3ms
 + anthropic==0.50.0
 + langchain-anthropic==0.3.12
⠧ Initializing UiPath project ...
✓   Created '.env' file.
✓   Created 'agent.mermaid' file.
✓   Created 'uipath.json' file.
🔧  Please ensure to define either ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file.
💡  Run agent: uipath run agent '{"topic": "UiPath"}'
```

This command creates the following files:

| File Name        | Description                                                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `.env`           | Environment variables and secrets (this file will not be packed & published).                                                     |
| `main.py`        | LangGraph agent code.                                                                                                             |
| `uipath.json`    | Input/output json schemas and bindings.                                                                                           |
| `langgraph.json` | [LangGraph](https://langchain-ai.github.io/langgraph/concepts/application_structure/#file-structure) specific configuration file. |
| `agent.mermaid`  | Graph visual representation.                                                                                                      |

## Set Up Environment Variables

Before running the agent, set either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the previously created `.env` file:

//// tab | Open AI

```hl_lines="3"
UIPATH_ACCESS_TOKEN=YOUR_TOKEN_HERE
UIPATH_URL=https://cloud.uipath.com/ACCOUNT_NAME/TENANT_NAME
OPENAI_API_KEY=sk-proj-......
```

////

//// tab | ANTHROPIC_API_KEY

```hl_lines="3"
UIPATH_ACCESS_TOKEN=YOUR_TOKEN_HERE
UIPATH_URL=https://cloud.uipath.com/ACCOUNT_NAME/TENANT_NAME
ANTHROPIC_API_KEY=your_api_key_here
```

////

## Run the Agent Locally

Execute the agent with a sample input:

<!-- termynal -->

```shell
> uipath run agent '{"topic": "UiPath"}'
[2025-04-29 12:31:57,756][INFO] ((), {'topic': 'UiPath'})
[2025-04-29 12:32:07,689][INFO] ((), {'topic': 'UiPath', 'report': "..."})
```

This command runs your agent locally and generates the report at standard output.

## Pack and Publish the agent on UiPath Cloud Platform

Follow these steps to publish and run your agent on UiPath Cloud Platform:

### Authenticate with UiPath

<!-- termynal -->

```shell
> uipath auth
⠋ Authenticating with UiPath ...
🔗 If a browser window did not open, please open the following URL in your browser: [LINK]
👇 Select tenant:
  0: Tenant1
  1: Tenant2
Select tenant number: 0
Selected tenant: Tenant1
✓  Authentication successful.
```

### (Optional) Update author details in `pyproject.toml`

```toml
authors = [{ name = "Your Name", email = "your.name@example.com" }]
```

### Package your project:

<!-- termynal -->

```shell
> uipath pack
⠋ Packaging project ...
Name       : test
Version    : 0.1.0
Description: Add your description here
Authors    : Your Name
✓  Project successfully packaged.
```

### Publish to your workspace

<!-- termynal -->

```shell
> uipath publish --my-workspace
⠙ Publishing most recent package: my-agent.0.0.1.nupkg ...
✓  Package published successfully!
⠦ Getting process information ...
🔗 Process configuration link: [LINK]
💡 Use the link above to configure any environment variables
```

> Please note that a process will be auto-created only upon publishing to **my-workspace** package feed.

### Configure environment variables using the provided link

![Set Environment Variables](quick_start_images/cloud_env_var.png)

## Invoke the Agent on UiPath Cloud Platform

Invoke the agent in UiPath Cloud Platform:

<!-- termynal -->

```shell
> uipath invoke agent '{"topic": "UiPath"}'
⠴ Loading configuration ...
⠴ Starting job ...
✨ Job started successfully!
🔗 Monitor your job here: [LINK]
```

Use the provided link to monitor your job and view detailed traces.

## Next steps

Congratulations! You have successfully set up, created, published, and run a UiPath LangChain Agent. 🚀

For more advanced agents and examples, please refer to our [samples section](https://github.com/UiPath/uipath-langchain-python/tree/main/samples).
