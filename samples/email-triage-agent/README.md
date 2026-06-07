# Email Triage Agent

A long-running LangGraph agent that watches a UiPath Integration Services Outlook connection for new emails matching a subject filter, classifies each one with an LLM, and replies to the original email with a polite acknowledgement drafted by the LLM.

The graph has no terminal node — once started, the agent stays SUSPENDED on the Outlook trigger forever, resuming to triage and reply to each matching email and then re-suspending. Cancel the job manually when you're done.

## What this sample demonstrates

- **`WaitIntegrationEvent`** — the agent suspends until an external IS connector event fires. The Connections-service registers a remote subscription on the user's behalf; when a matching email arrives, Orchestrator resumes the job and the SDK enriches the IS event metadata into the actual Microsoft Graph `Message`.
- An LLM call with strict structured output (Pydantic schema for the triage result).
- A direct Microsoft Graph call to send the reply, authenticated with the OAuth token issued for the same UiPath connection that received the trigger.

## Flow

```
START
  └─► wait_for_email      (suspend on Outlook EMAIL_RECEIVED, resume with Graph Message)
        └─► triage_email  (LLM → severity / category / summary / suggested_response)
              └─► send_reply  (Graph POST /me/messages/{id}/reply with the LLM draft)
                    └─► finalize  (log result, clear transient state, increment counter)
                          └─► wait_for_email  (loop)
```

## Input

The agent takes a single required input at job start:

```json
{ "subject": "Issue" }
```

| Field | Description |
|---|---|
| `subject` | Exact email subject to watch for. The IS trigger registers a server-side filter `(subject=='<value>')` so only matching emails fire it. |

`subject` is persisted in state across loop iterations — set once at job start.

## Connection (binding)

The Outlook connection used by the agent is declared as a **binding** in `bindings.json` rather than passed in as input. The code references a placeholder connection key (`<your-outlook-connection>`) and calls `sdk.connections.retrieve_async(OUTLOOK_CONNECTION_KEY)`. That method is decorated with `@resource_override("connection", resource_identifier="key")`, which inspects the runtime's binding-overwrite context and substitutes the deployer-selected connection's real key before the HTTP call. The agent then reads `connection.name` and `connection.folder.path` from the resolved connection and feeds them into `WaitIntegrationEvent`.

To run the agent:

- **In Orchestrator (deployed)**: pick the actual Outlook 365 connection when configuring the agent — Orchestrator's binding UI presents the deployer with the list of available `uipath-microsoft-outlook365` connections and overwrites the placeholder key.
- **Locally**: edit the `OUTLOOK_CONNECTION_KEY` constant at the top of `graph.py` to a real connection key in your tenant, or pass a resource-overwrites file via `--resource-overwrites`.

The connector key (`uipath-microsoft-outlook365`) is hardcoded.

The connection must be authorized to **read AND send** mail (`Mail.Read` + `Mail.Send` Graph scopes). Re-authorize the connection from the UiPath Connections UI if either scope is missing.

## Running locally

```bash
uv sync
uipath run agent '{"subject": "Issue"}'
```

The agent suspends waiting for the first matching email. Send (or have someone send) a message with subject `Issue` to the inbox the connection is bound to. When it arrives, the agent resumes, triages, replies, logs the result, and re-suspends on the next email.

Sample iteration log:

```
[INFO] Waiting for next email on '<your-outlook-connection>' (folder='<your-folder>') with subject='Issue' (triaged so far: 0)...
[INFO] Received email from alice@example.com: Issue
[INFO] Triage: severity=P0_critical category=bug
[INFO] Reply sent.
[INFO] Triaged email #1 from alice@example.com (subject='Issue', severity=P0_critical, category=bug, reply_sent=True)
[INFO] Waiting for next email on '<your-outlook-connection>' (folder='<your-folder>') with subject='Issue' (triaged so far: 1)...
```

## Notes

- **`subject` is set once at job start.** Persisted in state across loop iterations — to change it, cancel the job and start a new one. The connection is bound at deploy time via `bindings.json`, not via input.
- **Long-running pattern.** This sample is deliberately a single long-lived job to demo `WaitIntegrationEvent` cleanly. The idiomatic UiPath production pattern for "react to many emails" is the inverse: configure an Orchestrator event trigger that starts a fresh, one-shot agent job per matching email. That gives you a finite lifecycle per email, parallel processing, and no recursion-limit concerns. Use whichever shape fits your operational model.
