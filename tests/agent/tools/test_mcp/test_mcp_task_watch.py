"""Suspending the run on a task a Remote MCP server runs (``com.uipath/tasks``).

There is no Orchestrator job behind such a task, so the run asks the server
(AgentHub) to watch it, and suspends on an API trigger the server fires and on
a timer for when the watch ends. These tests drive the flow in a real graph
with a scripted call, and the one request that asks for a watch against a
recorded session.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from mcp.shared.exceptions import MCPError
from mcp.types import INTERNAL_ERROR, CallToolResult, TextContent
from uipath.core.feature_flags import FeatureFlags
from uipath.platform.common import WaitUntil
from uipath.platform.constants import ENV_FOLDER_KEY, ENV_JOB_KEY

from uipath_langchain.agent.tools.mcp.mcp_task_suspension import (
    API_SUSPENSION_FLAG,
    MAX_WATCHES_PER_CALL,
    SUSPENSION_KILL_SWITCH,
    UIPATH_TASK_META_KEY,
    UIPATH_TASKS_EXTENSION_ID,
    PollWindowElapsed,
    Submission,
    SuspensionSettings,
    TaskJob,
    TaskWatch,
    TaskWatchTerms,
    call_with_suspension,
    own_job,
    task_job,
    task_watch,
    task_watch_terms,
    tasks_capability,
)
from uipath_langchain.agent.tools.mcp.mcp_tasks import (
    McpTaskHandle,
    session_extension_kwargs,
    watch_claimed_task,
)

JOB_KEY = "5d7f1b2c-64a3-4e9b-8f2e-0c1d2e3f4a5b"
FOLDER_KEY = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
WATCH_ID = "3f2e1d0c-9b8a-4c7d-8e6f-5a4b3c2d1e0f"

#: No polling first, so a test reaches the watch without waiting.
SUSPEND_AT_ONCE = SuspensionSettings(min_polling_time_before_suspension=0)


@pytest.fixture(autouse=True)
def clean_flags() -> Any:
    FeatureFlags.reset_flags()
    yield
    FeatureFlags.reset_flags()


@pytest.fixture
def as_a_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run is an Orchestrator job, with suspending on a watch switched on."""
    monkeypatch.setenv(ENV_JOB_KEY, JOB_KEY)
    monkeypatch.setenv(ENV_FOLDER_KEY, FOLDER_KEY)
    FeatureFlags.configure_flags({API_SUSPENSION_FLAG: True})


def api_annotation(watch: Any = None, **overrides: Any) -> dict[str, Any]:
    """A task's ``_meta`` as AgentHub annotates a Remote server's task."""
    annotation: dict[str, Any] = {
        "version": 1,
        "suspendAfterMs": 60000,
        "resumeTrigger": {"triggerType": "Api"},
    }
    if watch is not None:
        annotation["watch"] = watch
    annotation.update(overrides)
    return {UIPATH_TASK_META_KEY: annotation}


# --- the annotation -------------------------------------------------------------


def test_an_api_annotation_says_the_task_can_be_watched() -> None:
    assert task_watch_terms(api_annotation()) == TaskWatchTerms(60.0)
    # It names no job, so the Job path leaves it alone.
    assert task_job(api_annotation()) is None


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {},
        {UIPATH_TASK_META_KEY: "yes"},
        api_annotation(version=2),
        api_annotation(version=True),
        api_annotation(resumeTrigger={"triggerType": "Job"}),
        api_annotation(resumeTrigger={"triggerType": "Timer"}),
        api_annotation(resumeTrigger="Api"),
    ],
)
def test_an_annotation_that_is_not_an_api_one_names_no_watch_terms(meta: Any) -> None:
    assert task_watch_terms(meta) is None


def test_the_watch_on_a_reply_is_read_with_its_lifetime() -> None:
    meta = api_annotation(watch={"id": WATCH_ID, "expiresInMs": 600000})

    assert task_watch(meta) == TaskWatch(WATCH_ID, 600.0)


@pytest.mark.parametrize(
    "watch",
    [
        None,
        "yes",
        {},
        {"id": WATCH_ID},
        {"id": "not-a-guid", "expiresInMs": 600000},
        {"id": WATCH_ID, "expiresInMs": 0},
        {"id": WATCH_ID, "expiresInMs": -5},
        {"id": WATCH_ID, "expiresInMs": "soon"},
        {"id": WATCH_ID, "expiresInMs": 10**400},
    ],
)
def test_a_watch_that_cannot_be_named_or_timed_is_not_suspended_on(watch: Any) -> None:
    assert task_watch(api_annotation(watch=watch)) is None


# --- what is declared -----------------------------------------------------------


def test_api_is_declared_only_by_a_job_that_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_JOB_KEY, raising=False)
    monkeypatch.delenv(ENV_FOLDER_KEY, raising=False)

    # By default: the Job kind only, as before.
    assert tasks_capability() == {"resumeTriggerTypes": ["Job"]}

    # Opted in, but not running as a job: there is nothing for a server to wake.
    FeatureFlags.configure_flags({API_SUSPENSION_FLAG: True})
    assert own_job() is None
    assert tasks_capability() == {"resumeTriggerTypes": ["Job"]}

    monkeypatch.setenv(ENV_JOB_KEY, JOB_KEY)
    monkeypatch.setenv(ENV_FOLDER_KEY, FOLDER_KEY)
    assert own_job() == (JOB_KEY, FOLDER_KEY)
    assert tasks_capability() == {"resumeTriggerTypes": ["Job", "Api"]}
    declared = session_extension_kwargs()["extensions"]
    assert declared[UIPATH_TASKS_EXTENSION_ID] == {"resumeTriggerTypes": ["Job", "Api"]}

    # The kill switch for suspension covers this kind too.
    FeatureFlags.configure_flags({SUSPENSION_KILL_SWITCH: True})
    assert tasks_capability() == {"resumeTriggerTypes": ["Job"]}


@pytest.mark.parametrize("job_key", ["", "not-a-guid", "4242"])
def test_a_job_key_that_is_not_a_guid_is_no_job_to_wake(
    monkeypatch: pytest.MonkeyPatch, job_key: str
) -> None:
    monkeypatch.setenv(ENV_JOB_KEY, job_key)
    monkeypatch.setenv(ENV_FOLDER_KEY, FOLDER_KEY)

    assert own_job() is None


# --- the flow, in a real graph ---------------------------------------------------


class WatchedCall:
    """A call to a Remote server whose task AgentHub can watch.

    Args:
        watches: What each ask for a watch answers, in order: a ``TaskWatch``
            (still working, watched), ``None`` (still working, unwatched) or a
            string (the task's result). The last one repeats.
        meta: The ``_meta`` on the task handle.
        finishes_in_window: The bounded wait finds the task finished.
    """

    def __init__(
        self,
        watches: list[Any],
        *,
        meta: Any = None,
        finishes_in_window: bool = False,
    ) -> None:
        self.meta = api_annotation() if meta is None else meta
        self.finishes_in_window = finishes_in_window
        self._watches = list(watches)
        self.submits = 0
        self.waits: list[tuple[float | None, bool]] = []
        self.asked: list[tuple[str, str]] = []

    async def submit(self) -> Submission[str, dict[str, Any]]:
        self.submits += 1
        return Submission(handle={"taskId": "task-1", "_meta": self.meta})

    async def wait(
        self, handle: dict[str, Any], *, seconds: float | None, poll_first: bool
    ) -> str:
        self.waits.append((seconds, poll_first))
        if seconds is not None and not self.finishes_in_window:
            raise PollWindowElapsed()
        return "the task's result"

    async def peek(self, handle: dict[str, Any]) -> str | None:
        raise AssertionError("a watched task is read by the ask for a watch")

    async def watch(
        self, handle: dict[str, Any], *, job_key: str, folder_key: str
    ) -> Any:
        assert handle["taskId"] == "task-1"
        self.asked.append((job_key, folder_key))
        return self._watches.pop(0) if len(self._watches) > 1 else self._watches[0]

    def watch_terms(self, handle: dict[str, Any]) -> TaskWatchTerms | None:
        return task_watch_terms(handle.get("_meta"))

    def dump_handle(self, handle: dict[str, Any]) -> dict[str, Any]:
        return dict(handle)

    def load_handle(self, saved: dict[str, Any]) -> dict[str, Any]:
        return dict(saved)

    def dump_result(self, result: str) -> Any:
        return {"text": result}

    def load_result(self, saved: Any) -> str:
        return saved["text"]

    def job(self, handle: dict[str, Any]) -> TaskJob | None:
        return task_job(handle.get("_meta"))


class _State(TypedDict):
    result: Any


def flow_graph(call: Any, settings: SuspensionSettings = SUSPEND_AT_ONCE) -> Any:
    async def node(state: _State) -> dict[str, Any]:
        return {
            "result": await call_with_suspension(
                call, tool_name="research", settings=settings
            )
        }

    builder = StateGraph(_State)
    builder.add_node("call", node)
    builder.add_edge(START, "call")
    builder.add_edge("call", END)
    return builder.compile(checkpointer=MemorySaver())


def thread(name: str) -> Any:
    return {"configurable": {"thread_id": name}}


def suspended_on(result: dict[str, Any]) -> tuple[str, dict[str, Any], WaitUntil]:
    """The one interrupt of a suspended run: an API request and a timer."""
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    api_request, timer = interrupts[0].value
    assert isinstance(timer, WaitUntil)
    return interrupts[0].id, api_request, timer


def woken(interrupt_id: str, by: Any = None) -> Command[Any]:
    """Resume as the runtime does, with whatever fired: a payload, or a timer."""
    return Command(
        resume={interrupt_id: by or {"com.uipath/task": {"watchId": WATCH_ID}}}
    )


async def test_a_working_task_is_suspended_on_through_a_watch(as_a_job: None) -> None:
    call = WatchedCall([TaskWatch(WATCH_ID, 600.0), "the task's result"])
    graph = flow_graph(call)
    before = datetime.now(timezone.utc)

    suspended = await graph.ainvoke({"result": None}, thread("w1"))

    interrupt_id, api_request, timer = suspended_on(suspended)
    # The API trigger's request carries the watch id: the server delivers only
    # to a trigger that does.
    assert api_request == {"com.uipath/task": {"watchId": WATCH_ID}}
    # The timer is the watch's end, so the run wakes whether or not it is woken.
    assert timedelta(seconds=590) < timer.resume_time - before < timedelta(seconds=610)
    # The server was asked to wake this run's own job, in its folder.
    assert call.asked == [(JOB_KEY, FOLDER_KEY)]

    final = await graph.ainvoke(woken(interrupt_id), thread("w1"))

    assert final["result"] == "the task's result"
    assert "__interrupt__" not in final
    # After the resume the task is read by a second ask, not polled, and the
    # call is never sent again.
    assert (call.submits, len(call.asked), call.waits) == (1, 2, [])


async def test_a_task_still_working_after_a_wake_is_watched_again(
    as_a_job: None,
) -> None:
    """A watch ends when its credentials would; a long task takes several."""
    second = "7c6b5a49-3827-4615-9f0e-1d2c3b4a5968"
    call = WatchedCall(
        [TaskWatch(WATCH_ID, 600.0), TaskWatch(second, 300.0), "the task's result"]
    )
    graph = flow_graph(call)

    first_suspend = await graph.ainvoke({"result": None}, thread("w2"))
    first_id, _, _ = suspended_on(first_suspend)
    # Woken by the timer: no payload from the server at all.
    second_suspend = await graph.ainvoke(woken(first_id, by="timer"), thread("w2"))

    second_id, api_request, _ = suspended_on(second_suspend)
    assert api_request == {"com.uipath/task": {"watchId": second}}
    assert second_id != first_id

    final = await graph.ainvoke(woken(second_id), thread("w2"))

    assert final["result"] == "the task's result"
    assert (call.submits, len(call.asked)) == (1, 3)


async def test_a_task_that_ends_while_polled_never_asks_for_a_watch(
    as_a_job: None,
) -> None:
    call = WatchedCall([TaskWatch(WATCH_ID, 600.0)], finishes_in_window=True)

    final = await flow_graph(call, SuspensionSettings(30)).ainvoke(
        {"result": None}, thread("w3")
    )

    assert final["result"] == "the task's result"
    assert "__interrupt__" not in final
    assert call.asked == []


async def test_a_task_found_finished_by_the_ask_is_returned(as_a_job: None) -> None:
    call = WatchedCall(["finished just now"])

    final = await flow_graph(call).ainvoke({"result": None}, thread("w4"))

    assert final["result"] == "finished just now"
    assert "__interrupt__" not in final


async def test_a_task_the_server_will_not_watch_is_polled_to_its_end(
    as_a_job: None,
) -> None:
    """A server with guardrails, a relay, a job it cannot see: no watch."""
    call = WatchedCall([None])

    final = await flow_graph(call).ainvoke({"result": None}, thread("w5"))

    assert final["result"] == "the task's result"
    assert "__interrupt__" not in final
    # One ask, then the ordinary unbounded poll.
    assert (len(call.asked), call.waits) == (1, [(None, False)])


async def test_outside_a_job_a_watchable_task_is_polled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_JOB_KEY, raising=False)
    FeatureFlags.configure_flags({API_SUSPENSION_FLAG: True})
    call = WatchedCall([TaskWatch(WATCH_ID, 600.0)])

    final = await flow_graph(call).ainvoke({"result": None}, thread("w6"))

    assert final["result"] == "the task's result"
    assert call.asked == []


async def test_without_the_opt_in_a_watchable_task_is_polled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_JOB_KEY, JOB_KEY)
    monkeypatch.setenv(ENV_FOLDER_KEY, FOLDER_KEY)
    call = WatchedCall([TaskWatch(WATCH_ID, 600.0)])

    final = await flow_graph(call).ainvoke({"result": None}, thread("w7"))

    assert final["result"] == "the task's result"
    assert "__interrupt__" not in final
    assert call.asked == []


async def test_switched_off_while_suspended_the_task_is_polled_after_the_wake(
    as_a_job: None,
) -> None:
    call = WatchedCall([TaskWatch(WATCH_ID, 600.0)])
    graph = flow_graph(call)
    suspended = await graph.ainvoke({"result": None}, thread("w8"))
    interrupt_id, _, _ = suspended_on(suspended)

    FeatureFlags.configure_flags({API_SUSPENSION_FLAG: False})
    final = await graph.ainvoke(woken(interrupt_id), thread("w8"))

    assert final["result"] == "the task's result"
    # No second ask, and the call is not sent again: the saved task is polled.
    assert (call.submits, len(call.asked), call.waits) == (1, 1, [(None, True)])


async def test_a_task_that_never_ends_stops_being_watched(as_a_job: None) -> None:
    call = WatchedCall([TaskWatch(WATCH_ID, 600.0)])
    graph = flow_graph(call)

    state = await graph.ainvoke({"result": None}, thread("w9"))
    for _ in range(MAX_WATCHES_PER_CALL - 1):
        interrupt_id, _, _ = suspended_on(state)
        state = await graph.ainvoke(woken(interrupt_id), thread("w9"))
    interrupt_id, _, _ = suspended_on(state)
    final = await graph.ainvoke(woken(interrupt_id), thread("w9"))

    # After the last watch it is polled, within the ordinary budget.
    assert final["result"] == "the task's result"
    assert len(call.asked) == MAX_WATCHES_PER_CALL
    assert call.waits == [(None, True)]


async def test_a_call_that_cannot_ask_for_a_watch_is_polled(as_a_job: None) -> None:
    """A client that carries the call without ``watch``: a coded agent's, today."""

    class Unwatchable(WatchedCall):
        watch = None  # type: ignore[assignment]

    call = Unwatchable([])

    final = await flow_graph(call).ainvoke({"result": None}, thread("w10"))

    assert final["result"] == "the task's result"
    assert call.waits == [(None, False)]


# --- the request that asks for a watch ------------------------------------------


class RecordedSession:
    """A session that answers one ``tasks/get`` and keeps what was sent."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.sent: list[dict[str, Any]] = []
        self.validated: list[str] = []

    async def send_request(self, request: Any, result_type: Any, **_: Any) -> Any:
        self.sent.append(
            request.model_dump(by_alias=True, mode="json", exclude_none=True)
        )
        if isinstance(self.answer, Exception):
            raise self.answer
        return result_type.model_validate(self.answer)

    async def validate_tool_result(self, tool_name: str, result: Any) -> None:
        self.validated.append(tool_name)


HANDLE = McpTaskHandle.model_validate({"taskId": "task-1", "status": "working"})


async def ask(session: RecordedSession) -> Any:
    return await watch_claimed_task(
        session,  # type: ignore[arg-type]
        "research",
        HANDLE,
        job_key=JOB_KEY,
        folder_key=FOLDER_KEY,
    )


async def test_the_ask_is_a_tasks_get_that_names_the_runs_own_job() -> None:
    session = RecordedSession(
        {
            "taskId": "task-1",
            "status": "working",
            "_meta": api_annotation(watch={"id": WATCH_ID, "expiresInMs": 600000}),
        }
    )

    watch = await ask(session)

    assert watch == TaskWatch(WATCH_ID, 600.0)
    assert session.sent == [
        {
            "method": "tasks/get",
            "params": {
                "taskId": "task-1",
                "_meta": {
                    "com.uipath/task": {
                        "watch": {"jobKey": JOB_KEY, "folderKey": FOLDER_KEY}
                    }
                },
            },
        }
    ]


async def test_a_working_task_with_no_watch_on_the_reply_answers_none() -> None:
    session = RecordedSession(
        {"taskId": "task-1", "status": "working", "_meta": api_annotation()}
    )

    assert await ask(session) is None


async def test_a_finished_task_answers_its_validated_result() -> None:
    session = RecordedSession(
        {
            "taskId": "task-1",
            "status": "completed",
            "result": {"content": [{"type": "text", "text": "done"}]},
        }
    )

    result = await ask(session)

    assert isinstance(result, CallToolResult)
    assert result.content == [TextContent(type="text", text="done")]
    assert session.validated == ["research"]


async def test_a_look_that_fails_for_now_answers_none_and_starts_no_suspend() -> None:
    session = RecordedSession(MCPError(INTERNAL_ERROR, "bad gateway"))

    assert await ask(session) is None


async def test_a_failed_task_is_raised_like_any_poll() -> None:
    session = RecordedSession(
        {"taskId": "task-1", "status": "failed", "statusMessage": "boom"}
    )

    with pytest.raises(MCPError):
        await ask(session)


# --- the interrupt, through the real trigger handler ----------------------------


async def test_the_suspend_becomes_an_api_trigger_and_a_timer() -> None:
    """Two sibling triggers; the first to fire resumes the run."""
    from uipath.core.triggers import UiPathResumeTriggerType
    from uipath.platform.resume_triggers import UiPathResumeTriggerHandler

    from uipath_langchain.agent.tools.mcp import mcp_task_suspension as suspension

    wait = suspension._wait_on_watch(TaskWatch(WATCH_ID, 600.0))

    triggers = await UiPathResumeTriggerHandler().create_triggers(wait)

    api, timer = triggers
    assert api.trigger_type == UiPathResumeTriggerType.API
    # The runtime mints the inbox id; the request is what the server reads to
    # find this run's trigger, so the watch id has to be in it.
    assert api.api_resume is not None and api.api_resume.inbox_id
    assert api.api_resume.request == {"com.uipath/task": {"watchId": WATCH_ID}}
    assert timer.trigger_type == UiPathResumeTriggerType.TIMER
    assert timer.resume_time is not None
