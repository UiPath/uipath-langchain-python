---
description: Investigate agent execution failure alerts end-to-end — triages, investigates representative failures in parallel, aggregates root causes, and creates Jira tickets for platform bugs
allowed-tools: Task, Bash, Read, Write, Edit, Grep, Glob, AskUserQuestion, mcp__atlassian__getJiraIssue, mcp__atlassian__getAccessibleAtlassianResources, mcp__atlassian__createJiraIssue, mcp__atlassian__addCommentToJiraIssue, mcp__atlassian__searchJiraIssuesUsingJql, mcp__atlassian__editJiraIssue
---

# Investigate Agent Execution Failure Alert

End-to-end investigation of an SRE alert containing multiple agent failures.
Triages the alert, investigates representative failures in parallel, aggregates
root causes, and creates Jira tickets for confirmed platform/SDK bugs.

## Usage

```
/investigate-alert <ticket> [--env stg|alp|prd] [--ago 24h|7d|...]
```

## Arguments

- `ticket` (required): Jira ticket ID (e.g., `SRE-536459`) or full Jira URL
- `--env` (optional): Override environment detection (stg, alp, prd)
- `--ago` (optional): Override time window (e.g., `48h`, `7d`). Default: auto-detected from alert + duplicate firings.

Parse the arguments from: $ARGUMENTS

## Orchestration Workflow

You are the orchestrator. You coordinate two types of subagents and aggregate their outputs
into a final report with actionable Jira tickets.

### Phase 1: Triage

Spawn a single **triage-alert** subagent. Tell it to read its own instructions from the
agent definition file — do NOT copy the file contents into the prompt.

```
Task(
  subagent_type="general-purpose",
  description="Triage alert <ticket>",
  prompt="Read your instructions from <repo>/.claude/agents/triage-alert.md and execute them.\n\nInputs:\n- TicketId: <ticket>\n- Environment: <env or 'auto-detect'>\n- Ago: <ago or 'auto-detect'>"
)
```

Wait for completion. The triage agent writes:
- `.ai-workspace/investigations/triage-<ticket>.md` — the triage report (human-readable)
- `.ai-workspace/investigations/triage-plan-<ticket>.json` — machine-readable investigation plan
- `.ai-workspace/investigations/alert-failures-<ticket>.json` — raw failure data

### Phase 2: Read Investigation Plan

After triage completes, read the **JSON plan file** directly:

```
Read(".ai-workspace/investigations/triage-plan-<ticket>.json")
```

This gives you structured data with no markdown parsing needed:
- `environment` / `envFlag` — environment name and PowerShell flag
- `ago` — time window to pass to investigation agents
- `investigate[]` — ALL categories to investigate, each with `jobKey`, `severity`, `signature`
- `deduplicated[]` — true duplicate categories (same error, same org, different wrapper) that
  are covered by another category's investigation

If the plan file is missing (triage agent wrote v1 format), fall back to reading the
triage markdown report and parsing the Recommended Investigation Plan table.

Investigate ALL categories in `investigate[]`. There is **no cap** — every non-deduplicated
category gets its own investigation agent. Even single-occurrence failures may turn out to be
real SDK bugs, so do not skip any.

### Phase 3: Parallel Investigation

For each investigation target, spawn an **investigate-job** subagent. Tell each subagent
to read its own instructions from the agent definition file — do NOT copy the file contents
into the prompt. Launch ALL of them in a single message to maximize parallelism.

```
Task(
  subagent_type="general-purpose",
  description="Investigate <signature-short>",
  prompt="Read your instructions from <repo>/.claude/agents/investigate-job.md and execute them.\n\nInputs:\n- JobKey: <jobKey>\n- Environment: <envFlag>\n- Ago: <ago>"
)
```

Wait for ALL investigation agents to complete. Each writes:
- `.ai-workspace/investigations/investigation-<jobkey>.md`

#### Failure Policy

- If an investigation agent fails or times out, record it as failed and continue with the rest.
- If **<=30%** of investigations fail: proceed normally, note failed categories in the report.
- If **>30%** of investigations fail: pause and ask the user whether to continue with partial
  results or abort. Something may be wrong with the data collection scripts or Azure access.

### Phase 4: Aggregate Results

Use the **subagent return summaries** (from the Task results) for initial aggregation.
Each investigation agent returns a concise summary with outcome, root cause category,
agent name/version, and top recommendations.

Only read the full investigation report file if:
- The summary is ambiguous about the root cause classification
- You need specific evidence details for a Jira ticket description
- You need to cross-reference findings across categories

For each investigated category, classify and prioritize:

#### Classify Each Root Cause

| Classification | Criteria | Action |
|---------------|----------|--------|
| **Platform Bug** | Bug in UiPath runtime, SDK, or platform services. Reproducible. Not caused by user configuration. | Create Jira ticket |
| **SDK Bug** | Bug in uipath-python, uipath-langchain, or uipath-agents packages | Create Jira ticket |
| **Infrastructure Issue** | Capacity, scaling, or infrastructure problem | Create Jira ticket if persistent |
| **User Misconfiguration** | Agent configured incorrectly by the user | No ticket — note in report |
| **External Dependency** | Third-party service issue (LLM provider, etc.) | No ticket unless UiPath should handle gracefully |
| **Transient** | One-off network/timing issue | No ticket |
| **Unknown** | Insufficient evidence to classify | Note in report, may need manual follow-up |

#### Determine Bug Priority

| Priority | Criteria |
|----------|----------|
| Blocker | Affects multiple organizations with no workaround, or causes data loss |
| Critical | Affects functionality for multiple orgs, workaround may exist |
| Major | Affects single org or limited scope, reproducible bug |
| Minor | Edge case, cosmetic, or has easy workaround |

### Phase 5: Duplicate Detection & Ticket Proposals

This phase is critical. Creating unnecessary tickets that get marked as duplicates wastes
engineering time and erodes trust in automated investigation. Be conservative: when in doubt,
do NOT create a ticket.

#### Step 5a: Search for Existing Tickets

For EACH category classified as Platform Bug or SDK Bug, search for potential duplicates.
Cast a wide net — false negatives (missing a duplicate) are much worse than false positives
(finding a non-duplicate match).

Run ALL 4 search strategies. For multiple proposed tickets, run ALL searches across ALL
tickets in a single parallel batch — do not search ticket-by-ticket sequentially.

Search strategies:

1. **Search by error type name** (the Python exception class):
   ```
   jql='text ~ "<innermostExceptionType>" AND statusCategory != Done ORDER BY created DESC'
   ```

2. **Search by key error message fragment** (strip UUIDs/IDs first, use the distinctive part):
   ```
   jql='text ~ "<distinctive error phrase>" AND statusCategory != Done ORDER BY created DESC'
   ```

3. **Search by affected service + HTTP status** (for EnrichedException patterns):
   ```
   jql='text ~ "<service_name>" AND text ~ "<status_code>" AND statusCategory != Done ORDER BY created DESC'
   ```

4. **Search the SRE project for recent similar alerts**:
   ```
   jql='project = SRE AND text ~ "<key phrase>" AND created >= -30d ORDER BY created DESC'
   ```

All searches use `cloudId="uipath.atlassian.net"` and `maxResults=10`.

For each search hit, read the ticket summary and description (fetch the issue if needed) to
determine if it genuinely covers the same root cause.

#### Step 5b: Classify Each Match

| Verdict | Meaning | Action |
|---------|---------|--------|
| **EXACT DUPLICATE** | An open ticket describes the identical root cause | SKIP — add a comment to the existing ticket instead with new evidence |
| **PARTIAL OVERLAP** | A ticket covers a related but different issue, or the same symptom but different root cause | Flag for user review |
| **STALE MATCH** | A ticket exists but is Done/Closed and the issue has regressed | Flag for user review — may need reopening |
| **NO MATCH** | No relevant existing ticket found across all searches | Candidate for creation, pending user approval |

#### Step 5c: Present Proposals to User

Do NOT create any tickets automatically. Present all proposals to the user using
AskUserQuestion and let them decide.

Show the user a summary for each proposed ticket with:
- Proposed ticket summary and priority
- Category signature and failure count
- Duplicate search results (what was found, verdict)
- Your recommendation (CREATE / SKIP / NEEDS REVIEW)

Example:
```
Proposed ticket: "UnicodeEncodeError in agent runtime when processing non-ASCII input"
Priority: Major
Evidence: 2 failures across 1 org, investigation confirmed ascii codec failure in SDK
Duplicate search:
  - "UnicodeEncodeError" -> 0 open matches
  - "ascii codec" -> 0 open matches
  - SRE project recent -> SRE-534210 (different: was about PDF encoding, not agent input)
Recommendation: CREATE — no existing ticket covers this
```

Use AskUserQuestion with options:
- "Create all recommended tickets"
- "Let me pick which ones to create"
- "Don't create any tickets — I'll handle manually"

If the user chooses to pick, present each ticket individually for approval.

#### Step 5d: Create Approved Tickets

Only after explicit user approval, create each ticket:

```
mcp__atlassian__createJiraIssue(
  cloudId="uipath.atlassian.net",
  projectKey="PC",
  issueTypeName="Bug",
  summary=<concise summary, max 120 chars>,
  description=<markdown description using template from .claude/templates/bug-ticket.md>
)
```

Use project key **PC** (`https://uipath.atlassian.net/jira/software/c/projects/PC`).

Read the ticket description template from `.claude/templates/bug-ticket.md`
and fill in the placeholders. If you need detailed evidence from an investigation report,
read the specific investigation file at this point.

#### Step 5e: Link & Comment on Existing Tickets

For categories where an existing ticket was found (EXACT DUPLICATE), add a comment to that
ticket with new evidence. Use this format:

```markdown
## Additional Evidence from <ticket>

This issue is still occurring. <N> new failures detected in the last <period>.

**Representative Job Key**: `<jobKey>`
**Affected Organizations**: <list>

<Brief root cause confirmation from investigation>
```

### Phase 6: Write Final Report

After all ticket decisions are finalized, write the aggregated report to
`.ai-workspace/investigations/alert-report-<ticket>.md`.

Read the report template from `.claude/templates/alert-report.md`
and fill in all sections. Include:
- Executive summary with classification breakdown
- Each investigated category with outcome, root cause, classification, priority, ticket action
- Deduplicated categories with which investigation covers them (from the triage plan's `deduplicated[]` array)
- Proposed and created tickets (with final verdicts)
- Cross-cutting observations
- Recommendations

Write the report **once** with all information, including ticket decisions from Phase 5.

### Phase 7: Comment on SRE Ticket

After the report is written, add a summary comment to the original SRE alert ticket:

```
mcp__atlassian__addCommentToJiraIssue(
  cloudId="uipath.atlassian.net",
  issueIdOrKey=<ticket>,
  commentBody=<markdown summary using template from .claude/templates/sre-comment.md>
)
```

Read the comment template from `.claude/templates/sre-comment.md`
and fill in the placeholders.

### Phase 8: Present Results

Show the user a concise summary:
- How many categories were investigated
- Ticket actions taken (created / updated existing / skipped)
- Top findings
- Path to the full report
- Any categories that need manual follow-up

## Guidelines

- **Subagent prompts must be lean.** Tell subagents to read their own instructions from file.
  Do NOT copy agent definition contents into the Task prompt. Each subagent has its own context
  window — let it spend its own tokens reading the file.
- **Spawn investigation agents in parallel** (single message with multiple Task calls).
- **Use subagent return summaries for aggregation.** Only read full investigation files when
  the summary is insufficient or you need specific evidence for a Jira ticket.
- If a subagent fails, log the error and continue with others. Report partial results.
- The triage agent already filtered out "User" ErrorCategory failures. All failures in the
  data are non-user errors, but investigation may still reveal user misconfiguration as the
  true root cause.
- All reports go in `.ai-workspace/investigations/`.
- **Templates live in `.claude/templates/`.** Read them when needed, don't memorize their contents.

### Ticket Creation Principles

- **Never create a ticket you aren't confident about.** If the investigation is inconclusive
  or the root cause is ambiguous, note it in the report and let the user decide.
- **Always search before creating.** Run all 4 search strategies. Read the top matches.
  A 2-minute search prevents a duplicate that wastes 30 minutes of engineering triage.
- **Always ask the user.** Present proposals with evidence and duplicate search results.
  The user makes the final call on every ticket.
- **Prefer updating existing tickets** over creating new ones. If an open ticket covers
  80% of the same issue, comment on it with new evidence rather than creating a near-duplicate.
- **Be specific in summaries.** "Agent fails with UnicodeEncodeError on non-ASCII input"
  is useful. "Agent execution failure" is not.
- **Include reproduction keys.** Every ticket must have a representative job key so
  engineers can pull the full investigation.
