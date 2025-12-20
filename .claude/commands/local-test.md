# Local Test Command

Run local tests on UiPath agent examples.

## Arguments
$ARGUMENTS

## Instructions

Parse the arguments to determine the action and example to run:

**Expected formats:**
- `/local-test run <example> [input]` - Run an agent with optional JSON input
- `/local-test eval <example>` - Run evaluations for an example
- `/local-test auth [env]` - Authenticate (alpha/staging/prod, default: alpha)
- `/local-test list` - List available examples
- `/local-test setup` - Show hacked-coded setup instructions

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

3. **auth**: Authenticate with UiPath
   - `alpha` (default): `uv run uipath auth --alpha`
   - `staging`: `uv run uipath auth --staging`
   - `prod`: `uv run uipath auth`

4. **list**: List examples by checking `examples/*/agent.json`

5. **setup**: Explain the hacked-coded setup process for using local editable dependencies

**Available examples:** calculator, basic, basic_with_ootb_guardrails, debug

Execute the appropriate command based on the parsed arguments. If arguments are empty or invalid, show usage help.
