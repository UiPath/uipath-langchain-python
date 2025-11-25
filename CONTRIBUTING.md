# Contributing to UiPath Agents Python

Most of the times when working on this repo, you'll also need to do some changes in `uipath-langchain-python` and/or `uipath-python`. In order to work with all 3 repositories at the same time, the standard way is to configure them as `editable` dependencies. You can start with just the `uipath-agents-python` project and continue with the setup for the other repositories when needed.

## Development Strategies

A simple way to make development easier and not combine code changes with agent defintitions, is to create a separate Python project that references the `uipath-agents-python` package as an editable dependency. For a better experience, start by creating a new root-level directory (e.g. `python-workspace`) that will hold all the related projects and repositories.

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

    # repeat the instructions to run an agent from [README.md](README.md)
    uv run uipath auth --alpha
    uv run uipath agent.json '{}'
    ```

3. You now have a fully functioning setup to start working on `uipath-agents-python`. Any change will be picked up by the `playground` project for testing.

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

2. **Update `pyproject.toml`** in `uipath-agents-python` and add the following section:

    ```toml
    [tool.uv.sources]
    uipath = { path = "../uipath-python", editable = true }
    uipath-langchain = { path = "../uipath-langchain-python", editable = true }
    ```

3. **Sync dependencies** for `uipath-agents-python`:

    ```bash
    uv sync
    ```

4. **Update `pyproject.toml`** in `uipath-langchain-python` and add the following section:

    ```toml
    [tool.uv.sources]
    uipath = { path = "../uipath-python", editable = true }
    ```

5. **Sync dependencies** for `uipath-langchain-python`:

    ```bash
    uv sync
    ```

6. You now have all the related repositories set up locally and linked for local development and direct feedback.

---

**IMPORTANT**: Do NOT commit the `editable` related changes to version control. This is for local development only. Using editable dependencies also updates the `uv.lock` files. If you need to update other dependencies in `pyproject.toml` files that also have `editable` dependencies, make sure to first revert the changes and do a clean `uv sync` to update the lockfile without the editable information and commit the changes before making them editable again.

## IDE Integrations

### VS Code Setup

A basic set of configuration options and recommended extensions can be found in [.vscode](.vscode). Feel free to disregared them if you already have your preferred setup.

The default python interpretor should already be loaded from the local `.venv`, but make sure to double check. This will probably be the main cause of problems when working cross-repostories. If the Intellisense is not working, check the configured interpretor.

### VS Code Debugging

A template is available in [launch.example.json](.vscode/launch.example.json). Copy it to `.vscode/launch.json` and update accordingly. This configuration is enough to debug any of the local repositories without additional setup.

Do note that there are 2 configurations: one for starting an agent and one for resuming an agent after an interrupt.

If you're using the `python-workspace` approach, you can copy the `.vscode` directory to the workspace and update the `cwd` property for the launch tasks to point to your agent's location (e.g. `${workspaceFolder}/playground/basic`).

### PyCharm Setup

Configure you Python interpreter to point to `.venv/bin/python`. Most of it's terminals will do some things automatically like activating the virtual environment if detected.

_More details to be added by someone using this._

#### Windows + WSL

This comes with the additional challange of adding the interpreter from the WSL instance. To my knowledge, this is only available in the Pro or higher version.

When adding the interpreter, choose the `On WSL` option, follow the wizard and choose the `Virtualenv Enviornment`.

### PyCharm Debugging

PyCharm unfortunately has it's own python debugging server and protocol and it does not play nice with others. Adding a direct start/resume configuration as we have for VS Code will not work.

1. Add a new configuration for `Python Debug Server`. You can keep the default IDE host name and ports. It'll give you some instructions on how to add the special pycharm debug package to you code and set it up.

    ```bash
    # 1. Add pydevd-pycharm ...:
    pip install pydevd-pycharm~=252.27397.106

    # 2. Add the following command to connect to the Debug Server:
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=5678, stdout_to_server=True, stderr_to_server=True)
    ```

2. Install the `pydevd-pycharm` package to the `uipath-agents-python` project using `uv`

    ```bash
    uv add --dev pydevd-pycharm~=252.27397.106
    ```

3. Add the python script to the top of a file that's used during execution (e.g. [cli_run.py](src/uipath_agents/_cli/cli_run.py))

    ```python
    1. import pydevd_pycharm # type: ignore
    2. pydevd_pycharm.settrace('localhost', port=5678, stdout_to_server=True, stderr_to_server=True)
    ```

4. Start the debug session in PyCharm; it will wait for the code to connect to the debug server

5. Run your code as you do normally

    ```bash
    uv run uipath agent.json '{}'
    ```

6. PyCharm should display a window to choose how to do path mapping. Using auto-detect with the 1st option should work fine.

#### Windows + WSL

If you're running you code in WSL, using `localhost` may not work to connect back to the PyCharm instance on Windows. If you get a connection refused error, replace `localhost` with the IP of the Windows host.

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
