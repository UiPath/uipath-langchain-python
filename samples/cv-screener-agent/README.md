# CV Screener Agent

A coded agent that screens a job application, guarded by five guardrails that read the
attached and produced files. It ports the low-code agent `CvScreenerGuardrailSolution`
from the Agents e2e suite: same system prompt, same Analyze Files and Create File tools,
same guardrails, with the rules, examples, thresholds and action reasons copied verbatim
from its `agent.json`. The guardrails are split between the two flavors, so one agent
shows both the `@guardrail` decorator and the middleware reading files.

The agent reads the CV with **Analyze Files**, returns a short summary and, when the
recruiter notes contain a code sample, saves it with **Create File** as `code_sample.py`
and returns that file in `codeSample`.

## Requirements

- Python 3.11+
- `uipath-langchain` 0.18.50 or newer. Run `uv sync` once that release is on PyPI.
- A tenant where the Agents guardrail file-support flags are on. Where they are off, a
  guardrail configured for files degrades to text.
- The judge model, `anthropic.claude-haiku-4-5-20251001-v1:0`, must be allowed for the
  LLM-as-judge guardrail by your governance policy.

## Guardrails

| Guardrail | Scope | Stage | Detection scope | Action | Flavor |
|---|---|---|---|---|---|
| Prompt injection in attached files | LLM | PRE | Files | Block | decorator: `@guardrail(PromptInjectionValidator(...))` on the LLM factory `make_llm` |
| Harmful content in attached files | Agent | PRE | Files | Block | middleware: `UiPathHarmfulContentMiddleware` |
| Candidate based outside the EU | Agent | PRE | Files | Log (Warning) | middleware: `UiPathLLMAsJudgeMiddleware` |
| Only CVs and cover letters | Tool (Analyze Files) | PRE | Files | Block | decorator: `guardrail(LLMAsJudgeValidator(...))` on the Analyze Files tool object |
| Protected code in output files | Agent | POST | Files | Block | middleware: `UiPathIntellectualPropertyMiddleware` |

The Analyze Files tool has no synchronous implementation, and `create_agent` calls it
through `ainvoke`. The decorator guards `ainvoke` as well as `invoke`, so it checks every
Analyze Files call.

Analyze Files reads the file through `UiPathChatOpenAI`; the normalized `UiPathChat`
client does not yet send file content blocks to the provider. The agent's own model never
receives file content, so it stays on `UiPathChat`.

## How the file reaches the guardrails

The input is `file: Attachment`, which the runtime exposes as a `job-attachment`. The
`screen_cv` node renders it into the prompt with `render_attachments_block`:

```text
Summarise the candidate's CV in the attached file using the Analyze Files tool.
<uip:attachments>[{"ID": "…", "FullName": "cv_non_eu_candidate.pdf", "MimeType": "application/pdf"}]</uip:attachments>
Recruiter notes: …
```

The model reads the block and copies the object into the `attachments` argument of
Analyze Files. The guardrails parse the same block into the run's attachment registry:

- the agent input checks and the LLM check forward every file the run holds;
- the Analyze Files check forwards the files its arguments mention, with the name and
  type each mention carries, without checking that the run holds them (the
  `PARITY_CHECK` middleware twin forwards only the files the run holds);
- the agent output check forwards only the files the result references, here
  `codeSample` in the structured response.

Only references (`id`, `fileName`, `mimeType`) are sent. The Agents backend downloads the
file, decodes or extracts its text and evaluates it. Analyze Files reads the file with its
own model instance, so the LLM check runs on the agent's model calls only, as in the
low-code agent.

## Run it

```bash
uv sync
uv run uipath auth
```

`uipath auth` writes the tenant URL and an access token to `.env`.

Upload a file as an Orchestrator attachment with `upload_attachment.py`. It reads the
credentials from `.env`, uploads each file you pass, and prints its attachment id together
with the `uipath run` command that screens it:

```bash
uv run python upload_attachment.py fixtures/cv_prompt_injection.txt
```

```text
cv_prompt_injection.txt  ->  f9674749-41e1-49cf-3d73-08df270eb7c1
uv run uipath run agent '{"file": {"ID": "f9674749-41e1-49cf-3d73-08df270eb7c1", "FullName": "cv_prompt_injection.txt", "MimeType": "text/plain"}, "recruiterNotes": ""}'
```

Run the printed command. The script:

- sets `MimeType` from the file extension (`.txt`, `.md`, `.csv`, `.json`, `.pdf`, `.png`,
  `.jpg`, `.docx`, `.xlsx`), so the guardrails know how to read the file;
- accepts several files and prints one command per file;
- uploads to the folder in your SDK configuration (`UIPATH_FOLDER_PATH` or
  `UIPATH_FOLDER_KEY`) when one is set;
- with `--refresh-token`, first runs `uip login refresh` (the `uip` CLI) and writes the
  new access token to `.env`. Without the `uip` CLI, run `uv run uipath auth`
  again when the token expires.

Upload a file again for every run you want to compare: the Agents backend caches a file's
content per attachment id for five minutes.

## Expected outcomes

Each fixture carries one trigger. The guardrails that run before it pass on the file.

| Input | Outcome |
|---|---|
| `cv_prompt_injection.txt` | Blocked by *Prompt injection in attached files* at the LLM check, before the first model call. |
| `cv_harmful_content.pdf` | Blocked by *Harmful content in attached files* at agent input. The model is never called. |
| `recipe_not_a_cv.md` | The Analyze Files call is blocked by *Only CVs and cover letters*. The tool never runs. |
| `cv_non_eu_candidate.pdf` | Completes with a summary. *Candidate based outside the EU* logs a warning: `[GUARDRAIL] [Candidate based outside the EU] Failed: …`. |
| `cv_non_eu_candidate.pdf` with a code sample in `recruiterNotes` | The model saves `code_sample.py` with Create File and returns it in `codeSample`. Blocked by *Protected code in output files* at agent output. |

For the last row, put the code in `recruiterNotes` with each line break written as `\n`;
the e2e test uses the Azure AI Content Safety protected-code sample, a pygame loop.

*Protected code in output files* depends on AL-611 (IP reads output files) in two ways:

- An agent output check forwards only the files the result references. The middleware
  reads them from the structured response, which is how `codeSample` reaches it.
- The Agents backend must read the `.py` file by its content, which the AL-611 backend
  change adds. On a tenant without it, the check does not read the file.

Outside a job, for example a plain local `uipath run`, Create File keeps the file in local
temporary storage instead of Orchestrator, so the backend cannot read it. Run the agent as
a job, for example a debug run on a robot, to exercise this row.

## Switches

Both are read when `graph.py` is imported.

### `PARITY_CHECK=1`

Adds the other flavor of each guardrail, with the flavor in its name:

| Added guardrail | Flavor |
|---|---|
| Prompt injection in attached files (middleware) | `UiPathPromptInjectionMiddleware` |
| Only CVs and cover letters (middleware) | `UiPathLLMAsJudgeMiddleware(scopes=[TOOL], tools=[analyze_files])` |
| Harmful content in attached files (decorator) | `@guardrail(HarmfulContentValidator(...))` on the agent factory `make_agent` (Agent scope) |
| Candidate based outside the EU (decorator) | `@guardrail(LLMAsJudgeValidator(...))` with `LogAction` on `make_agent` |
| Protected code in output files (decorator) | `@guardrail(IntellectualPropertyValidator(...), stage=POST)` on `make_agent` |

Both members of a pair send the same `attachments`, `detectionScope` and validator. The
decorators on `make_agent` run before the middleware's agent hooks, the prompt-injection
middleware runs before the decorated model, and the only-CVs middleware runs before the
decorated tool. The first member of a pair that sees a violation blocks the run, so the
other one is not called.

### `GUARDRAILS_DETECTION_SCOPE`

`Text`, `Files` or `Both` (default `Files`). Overrides `detection_scope` on every guardrail.
`Text` is the control run: no file reaches a guardrail, so they judge only the prompt text
and the tool arguments. The only-CVs judge still sees `recipe_not_a_cv.md` in the Analyze
Files arguments, so the recipe fixture may be blocked on its file name even under
`GUARDRAILS_DETECTION_SCOPE=Text`; the control claim holds for the other fixtures.

See [docs/guardrails.md](../../docs/guardrails.md) for how files reach coded guardrails.
