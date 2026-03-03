# Contributing to UiPath Agents Python

Most of the time when working on this repo, you'll also need to do some changes in `uipath-langchain-python` and/or `uipath-python`. In order to work with all 3 repositories at the same time, the standard way is to configure them as `editable` dependencies. You can start with just the `uipath-agents-python` project and continue with the setup for the other repositories when needed.

## Development Strategies

A simple way to make development easier and not combine code changes with agent definitions, is to create a separate Python project that references the `uipath-agents-python` package as an editable dependency. For a better experience, start by creating a new root-level directory (e.g. `python-workspace`) that will hold all the related projects and repositories.

```bash
# . python-workspace/
# └── uipath-agents-python/
```

You can open this directory in your IDE to make your life easier instead of opening 3+ instances.

### Dedicated Playground Project

1. Create and initialize a new Python project in the root workspace.

    ```bash
    # in [python-workspace]
    mkdir playground
    cd playground
    uv init
    uv add --editable ../uipath-agents-python # this will also do a uv sync
    ```

    ```bash
    # Your workspace should look similar to this

    # . python-workspace/
    # ├─┬ playground/
    # | ├── <other files>
    # | ├── pyproject.toml
    # | └── uv.lock
    # └── uipath-agents-python/
    ```

2. Add your agent definitions to the `playground` project. We'll start by copying the basic example.

    ```bash
    # in [playground]
    cp -R ../uipath-agents-python/examples/basic basic
    cd basic

    # Ensure agent.json is at the top level (required for runtime detection)
    ls agent.json

    # Authenticate and run
    uv run uipath auth --alpha
    uv run uipath run agent.json '{}'
    ```

3. You now have a fully functioning setup to start working on `uipath-agents-python`. Any change will be picked up by the `playground` project for testing.

**Key behavior to remember:**
- The runtime automatically detects agents by the presence of `agent.json` at the top level - this MUST be present at all times even if `.agent-builder/` with `agent.json` in already exists.
- If you create a `.agent-builder/` directory, it would override the top-level `agent.json` and `bindings.json` during execution.
- Copying contents of .agent-builder runs during both `run` and `debug` commands and files are copied only if the directory exists, without being a critical dependency.

### Multiple Repositories

Due to how the SDKs are structured, most of the code used for agents in stored in `uipath-langchain-python`. And in order to test the changes locally, you'll also need to set it up as an `editable` dependency. While we're at it, we'll do the same for `uipath-python` to have a complete development environment.

---

**IMPORTANT**: Using the local repositories is a requirement when updating related code, but it introduces some overhead:

-   You first need to make sure that you're on the latest version of any of the related packages before commiting any changes.
-   Update the project's version in the `uipath-agents-python` (e.g. `uipath-langchain-python>=<the version from pyproject.toml>`) if outdated.

-   There can be changes on related packages that are have not been yet published to the official feed; in this case, the published agents package will not use the same code as it does during local development; if you rely on those specific changes, they need to be published first.

---

1. **Clone the repositories** in the root workspace ([related links](#related-projects)):

    ```bash
    # in [python-workspace]
    git clone <uipath-langchain-python-url>
    git clone <uipath-python-url>

    # Your directory structure should look like:
    # . python-workspace
    # ├── playground/
    # ├── uipath-agents-python/
    # ├── uipath-langchain-python/
    # └── uipath-python/
    ```

2. **Configure Playground Project** -- Add your agent definitions to the playground project (see [Dedicated Playground Project](#dedicated-playground-project))

3. **Configure editable dependencies** - Choose one of the following approaches:

    #### Option A: Automated Setup (Recommended)

    Copy the setup scripts from `uipath-agents-python/.pycharm/` to your `python-workspace` directory and run the setup script:

    ```bash
    # in [python-workspace]
    cp uipath-agents-python/.pycharm/setup-local-dev.sh .
    cp uipath-agents-python/.pycharm/clean-local-dev.sh .
    chmod +x setup-local-dev.sh clean-local-dev.sh
    ./setup-local-dev.sh
    ```

    The script will automatically:
    - Add `[tool.uv.sources]` sections to the required `pyproject.toml` files
    - Sync dependencies for both `uipath-agents-python` and `uipath-langchain-python`

    To clean up the editable dependencies later (e.g., before committing changes), run:

    ```bash
    # in [python-workspace]
    ./clean-local-dev.sh
    ```

    #### Option B: Manual Setup

    If you prefer to configure manually:

    1. **Update `pyproject.toml`** in `uipath-agents-python` and add the following section:

        ```toml
        [tool.uv.sources]
        uipath = { path = "../uipath-python", editable = true }
        uipath-langchain = { path = "../uipath-langchain-python", editable = true }
        ```

    2. **Sync dependencies** for `uipath-agents-python`:

        ```bash
        cd uipath-agents-python
        uv sync
        ```

    3. **Update `pyproject.toml`** in `uipath-langchain-python` and add the following section:

        ```toml
        [tool.uv.sources]
        uipath = { path = "../uipath-python", editable = true }
        ```

    4. **Sync dependencies** for `uipath-langchain-python`:

        ```bash
        cd uipath-langchain-python
        uv sync
        ```

4. You now have all the related repositories set up locally and linked for local development and direct feedback.

---

**IMPORTANT**: Do NOT commit the `editable` related changes to version control. This is for local development only. Using editable dependencies also updates the `uv.lock` files. If you need to update other dependencies in `pyproject.toml` files that also have `editable` dependencies, make sure to first revert the changes and do a clean `uv sync` to update the lockfile without the editable information and commit the changes before making them editable again.

## IDE Integrations

### VS Code Setup

A basic set of configuration options and recommended extensions can be found in [.vscode](.vscode). Feel free to disregard them if you already have your preferred setup.

The default python interpreter should already be loaded from the local `.venv`, but make sure to double check. This will probably be the main cause of problems when working cross-repositories. If the Intellisense is not working, check the configured interpreter.

We recommend using a vs-code [multi-root workspace](https://code.visualstudio.com/docs/editing/workspaces/multi-root-workspaces)
which will allow the mypy (type checking) and ruff (lint/formatting) plugins to find and use the config for each individual package.
Without this, type checking in vscode will report invalid errors.

To use a multi-root workspace, create an `ur.code-workspace` file in your workspace root, then open that file in vscode:

```
{
  "folders": [
    { "path": "uipath-langchain-python", "name": "uipath-langchain" },
    { "path": "uipath-python", "name": "uipath" },
    { "path": "uipath-runtime-python", "name": "uipath-runtime" },
    { "path": "uipath-agents-python", "name": "uipath-agents" },
    { "path": "uipath-llamaindex-python", "name": "uipath-llamaindex" },
    { "path": "uipath-mcp-python", "name": "uipath-mcp" },
    { "path": "uipath-robot-python", "name": "uipath-robot" },
    { "path": "uipath-dev-python", "name": "uipath-dev" },
    { "path": "uipath-core-python", "name": "uipath-core" },
    { "path": "playground", "name": "playground" }
  ],
  "settings": {
    "python.analysis.diagnosticMode": "workspace",
    "mypy-type-checker.importStrategy": "fromEnvironment"
  }
}
```

### VS Code Debugging

A template is available in [launch.example.json](.vscode/launch.example.json). Copy it to `.vscode/launch.json` and update accordingly. This configuration is enough to debug any of the local repositories without additional setup.

Do note that there are 2 configurations: one for starting an agent and one for resuming an agent after an interrupt.

If you are using a multi-root workspace, you can copy the `.vscode` directory to your playground directory and update the `cwd` property for the
launch tasks to point to your agent's location (e.g. `${workspaceFolder}/basic`). Because your using a multi-root
workspace, the ${workspaceFolder} will be replaced by the package folder, not the workspace root folder.

#### Windows + WSL

This comes with the additional challange of adding the interpreter from the WSL instance. To my knowledge, this is only available in the Pro or higher version.

When adding the interpreter, choose the `On WSL` option, follow the wizard and choose the `Virtualenv Enviornment`.

### PyCharm Setup

This setup is strongly recommended with [Multiple Repositories setup](#multiple-repositories).

1. **Create a PyCharm workspace**:
   - Open PyCharm and create a new workspace by selecting the root folder (`python-workspace`)
   - The workspace should include all 4 folders: `playground`, `uipath-agents-python`, `uipath-langchain-python`, and `uipath-python`

2. **Configure the Python interpreter**:
   - Set your Python interpreter to point to `playground/.venv/bin/python` (or `playground\.venv\Scripts\python.exe` on Windows)
   - PyCharm terminals will automatically activate the virtual environment when detected

3. **Copy PyCharm configuration files**:
   - Copy the entire runConfigurations directory from `uipath-agents-python/.pycharm/runConfigurations` into your `python-workspace/.idea` (`cp -R .pycharm/runConfigurations .idea/`)
   - Restart PyCharm and check if run configurations are visible
   - If your agent requires input parameters, change configuration script parameters to the format where strings are escaped: `run agent.json {\"word\":\"donkey\"}`

4. **Set up editable dependencies** (if working with multiple repositories):
   - Use the setup scripts to automatically configure the local development environment (see [Multiple Repositories](#multiple-repositories) section for details)
   - Alternatively, follow the manual setup instructions in that section

### PyCharm Debugging

Once the workspace is configured with all repositories and editable dependencies are set up, debugging works seamlessly across all projects:

- **Set breakpoints** anywhere in the code across `uipath-agents-python`, `uipath-langchain-python`, or `uipath-python`
- **Run any configuration in debug mode** - PyCharm will stop at breakpoints in any of the three repositories
- **Note**: This requires that the `pyproject.toml` files are configured to use local editable versions of the libraries (see [Multiple Repositories](#multiple-repositories) section)

#### Windows + WSL

If you're running you code in WSL, using `localhost` may not work to connect back to the PyCharm instance on Windows. If you get a connection refused error, replace `localhost` with the IP of the Windows host.

## Pulling Agent Projects (Local Development Only)

The `uipath pull` command can be extended with agent-specific support through a middleware entry point. This is **only available during local development** and is not part of the published package.

### What It Does

The pull middleware adds the capability to `uipath pull`:
- **Studio Web mode** — Detects `agent.json` in the remote project and pulls all agent project files (instead of treating it as a coded project).

### Setup

The middleware entry point is registered automatically by the `setup-local-dev.sh` script (see [Multiple Repositories - Option A](#option-a-automated-setup-recommended)). It adds the following to `pyproject.toml`:

```toml
[project.entry-points."uipath.middlewares"]
agents_middleware = "uipath_agents.middlewares:register_middleware"
```

If you're using the manual setup, add this section to `uipath-agents-python/pyproject.toml` before `[project.urls]` and run `uv sync`.

> **Do NOT commit this entry point** — it is for local development only. The `clean-local-dev.sh` script will remove it along with the editable source configuration.

### Usage

```bash
# Pull an agent project from Studio Web (requires UIPATH_PROJECT_ID)
uv run uipath pull
```

```bash
# Pull an agent project from Studio Web in a passed directory
uv run uipath pull ./tmp/agent-project
```

**Prerequisites**: You must be authenticated (`uv run uipath auth --alpha`) and have `UIPATH_URL` set. Studio Web mode also requires `UIPATH_PROJECT_ID` — add it to the `.env` file in your agent project directory:

```env
UIPATH_PROJECT_ID=<your-project-id>
```

Alternatively, you can export it as an environment variable: `export UIPATH_PROJECT_ID=<your-project-id>`.

## Code Quality Tools

### Formatting & Linting (Ruff)

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Link and auto-fix issues
uv run ruff check . --fix
```

### Type Checking (mypy)

```bash
# Check entire codebase
uv run mypy .

# Check specific file
uv run mypy src/uipath_agents/agent_graph_builder/graph.py
```

## Running Tests

TBA

## CI/CD - GitHub Actions + Azure Pipelines

### PR Checks

### alpha/main version publish

### production version publish

## Related Projects

-   [uipath-python](https://github.com/UiPath/uipath-python) - Core UiPath Python SDK
-   [uipath-langchain-python](https://github.com/UiPath/uipath-langchain-python) - UiPath LangChain Integration SDK
