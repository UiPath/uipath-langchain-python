# Local Test Command

Run local tests on UiPath agent examples.

## Arguments
$ARGUMENTS

## Instructions

Parse the arguments to determine the action and example to run:

**Expected formats:**
- `/local-test run <example> [input]` - Run an agent with optional JSON input
- `/local-test eval <example>` - Run evaluations for an example
- `/local-test debug <example>` - Debug an agent interactively
- `/local-test auth [env]` - Authenticate (alpha/staging/prod, default: alpha)
- `/local-test e2e [test-file]` - Run e2e tests (requires env vars)
- `/local-test list` - List available examples
- `/local-test setup` - Show setup instructions

**Actions:**

1. **run**: Execute the agent
   ```bash
   cd examples/<example> && uv run uipath run agent.json '<input>'
   ```
   If no input provided, run without input:
   ```bash
   cd examples/<example> && uv run uipath run agent.json
   ```

2. **eval**: Run evaluations
   ```bash
   cd examples/<example> && uv run uipath eval agent.json
   ```

3. **debug**: Debug agent interactively
   ```bash
   cd examples/<example> && uv run uipath debug agent.json
   ```

4. **auth**: Authenticate with UiPath
   - Interactive auth:
     - `alpha` (default): `uv run uipath auth --alpha`
     - `staging`: `uv run uipath auth --staging`
     - `prod`: `uv run uipath auth`
   - Client credentials auth (for CI/automation):
     ```bash
     uv run uipath auth --client-id="$UIPATH_CLIENT_ID" --client-secret="$UIPATH_CLIENT_SECRET" --base-url="$UIPATH_URL"
     ```

5. **e2e**: Run e2e tests
   - Run all e2e tests: `uv run pytest tests/e2e/ -v -m "e2e"`
   - Run specific test file: `uv run pytest tests/e2e/test_run.py -v -m "e2e"`
   - Run without slow tests: `uv run pytest tests/e2e/ -v -m "e2e and not slow"`

   Required environment variables:
   - `UIPATH_CLIENT_ID`: Client ID for authentication
   - `UIPATH_CLIENT_SECRET`: Client secret for authentication
   - `UIPATH_URL`: Base URL (e.g., https://alpha.uipath.com/org/tenant)
   - `UIPATH_PROJECT_ID`: Project ID (required for eval tests)

6. **list**: List examples by checking `examples/*/agent.json`

7. **setup**: Explain the setup process for using local editable dependencies

**Available examples:** calculator, basic, basic_with_ootb_guardrails, debug

**Examples with evaluations:** calculator

Execute the appropriate command based on the parsed arguments. If arguments are empty or invalid, show usage help.
