# Quickstart Guide: UiPath LangChain Agents

## Introduction

This guide walks you through the process of setting up, creating, publishing, and running your first UiPath-LangChain Agent.

## Prerequisites

Ensure you have the following installed before proceeding:

-   Python 3.10 or higher
-   `pip` or `uv` package manager
-   A UiPath Platform account with appropriate permissions

## Step-by-Step Guide

## Creating a New Project

We recommend using `uv` for package management. To create a new project:

```shell
mkdir example
cd example
uv init . --python 3.10
```
This command creates a basic project structure.

## Install the UiPath LangChain SDK

Add the UiPath LangChain SDK to your project:

```bash
uv add uipath[langchain]
```
To verify the installation, run:

```shell
 uv run uipath -lv
```

## (Optional) Activate your virtual environment
```shell
# Windows
.venv\Scripts\activate
```
or
```shell
# Unix-like Systems
source .venv/bin/activate
```
> If you skip activating the virtual environment, prefix the upcoming commands with `uv run`.


## Create Your First UiPath Agent

Generate your first UiPath LangChain:

```bash
uipath new my-agent
```
This command creates the following files:

| File Name        | Description                                                                                                                       |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `.env`           | Environment variables and secrets (this file will not be packed & published).                                                     |
| `main.py`        | LangGraph agent code.                                                                                                             |
| `uipath.json`    | Input/output json schemas and bindings.                                                                                           |
| `langgraph.json` | [LangGraph](https://langchain-ai.github.io/langgraph/concepts/application_structure/#file-structure) specific configuration file. |
| `agent.mermaid`  | Graph visual representation.                                                                                                      |

## Set Up Environment Variables

Before running the agent, set either `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in the previously created `.env` file:

```
ANTHROPIC_API_KEY=your_api_key_here
```
or
```
OPENAI_API_KEY=your_api_key_here
```

## Run the Agent Locally

Execute the agent with a sample input:

```bash
uipath run agent '{"topic": "UiPath"}'
```

This command runs your agent locally and generates the report at standard output.

## Pack and Publish the agent on UiPath Cloud Platform

Follow these steps to publish and run your agent on UiPath Cloud Platform:

### Authenticate with UiPath
```bash
uipath auth
```

### (Optional) Update author details in `pyproject.toml`
```toml
authors = [{ name = "John Doe", email = "john.doe@example.com" }]
```

### Package your project:
```bash
uipath pack
```

### Publish to your workspace
```bash
uipath publish --my-workspace
```
> Please note that a process will be auto-created only upon publishing to **my-workspace** package feed.

### Configure environment variables using the provided link

![Set Environment Variables](quick_start_images/cloud_env_var.png)

## Invoke the Agent on UiPath Platform

Invoke the agent in UiPath Cloud Platform:

```bash
uipath invoke agent '{"topic": "UiPath"}'
```

**Expected output:**
```
⠦ Loading configuration ...
⠦ Starting job ...
✨ Job started successfully!
🔗 Monitor your job here: https://example.com
```

Use the provided link to monitor your job and view detailed traces.

## Next steps

Congratulations! You have successfully set up, created, published, and run a UiPath LangChain Agent. 🚀

For more advanced agents and examples, please refer to our [samples section](../samples).
