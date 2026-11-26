"""Suspending the run on a long MCP task (``com.uipath/tasks``).

Three layers, each against as much of the real thing as it can reach:

- the job annotation and the polling-time setting, as pure functions;
- the suspend flow inside a real LangGraph graph with a checkpointer, driven by
  a scripted call, so that a suspend is a real ``GraphInterrupt`` and a resume a
  real replay of the node;
- the MCP tool in its own tool node against a real HTTP server, resumed by a
  *second* client on a new connection -- which is what an Orchestrator resume
  is: a new process holding nothing but the checkpoint.
"""

import json
import time
from typing import Any, TypedDict
from unittest.mock import AsyncMock, Mock

import anyio
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.messages.content import create_tool_call
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from starlette.responses import Response
from uipath.core.feature_flags import FeatureFlags
from uipath.platform.common import WaitJobRaw
from uipath.platform.orchestrator.job import Job

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.react.agent import create_agent
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.mcp import create_mcp_tools
from uipath_langchain.agent.tools.mcp import mcp_task_suspension as suspension
from uipath_langchain.agent.tools.mcp.mcp_task_suspension import (
    MIN_POLLING_TIME_FLAG,
    SUSPENSION_KILL_SWITCH,
    UIPATH_TASK_META_KEY,
    UIPATH_TASKS_EXTENSION_ID,
    PollWindowElapsed,
    Submission,
    SuspensionSettings,
    TaskJob,
    call_with_suspension,
    polling_time_before_suspension,
    suspension_settings,
    task_job,
)
from uipath_langchain.agent.tools.tool_node import create_tool_node

from .real_server import (
    PINNED_TOOL_NAME,
    PinnedVersionServer,
    RecordingGateway,
    make_client,
    make_resource_config,
    patched_sdk,
    serve,
)
from .test_mcp_tasks_real_http import TaskServingServer


class _StubChatAnthropic(ChatAnthropic):
    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


JOB_KEY = "2be27a69-7a29-4793-b4d7-d50b740a4efa"
FOLDER_KEY = "c13c726f-24c2-47a4-a4b1-a6b9273afe36"

#: Poll first, as the default does -- for a moment rather than its minute.
POLLING_TIME = 0.3
POLL_FIRST = SuspensionSettings(min_polling_time_before_suspension=POLLING_TIME)

#: Suspend without polling first.
SUSPEND_AT_ONCE = SuspensionSettings(min_polling_time_before_suspension=0)


@pytest.fixture(autouse=True)
def clean_flags() -> Any:
    FeatureFlags.reset_flags()
    yield
    FeatureFlags.reset_flags()


def annotation() -> dict[str, Any]:
    """A task's ``_meta`` as a ``com.uipath/tasks`` server sends it."""
    return {
        UIPATH_TASK_META_KEY: {
            "version": 1,
            "jobKey": JOB_KEY,
            "folderKey": FOLDER_KEY,
            "resumeTrigger": {"triggerType": "Job", "itemKey": JOB_KEY},
        }
    }


# --- the job annotation and the setting ----------------------------------------


def test_the_annotation_is_read_into_a_job() -> None:
    assert task_job(annotation()) == TaskJob(JOB_KEY, FOLDER_KEY)


@pytest.mark.parametrize(
    "overrides",
    [
        {"resumeTrigger": {"triggerType": "Api", "inboxId": JOB_KEY}},
        {"resumeTrigger": {"triggerType": "job"}},
        {"resumeTrigger": {}},
        {"resumeTrigger": "Job"},
        {"resumeTrigger": None},
        {"version": 2},
        {"version": "1"},
        {"version": True},
        {"version": None},
    ],
    ids=[
        "another-trigger-kind",
        "trigger-kind-in-another-case",
        "no-trigger-kind",
        "trigger-not-an-object",
        "no-trigger",
        "a-later-version",
        "version-a-string",
        "version-a-boolean",
        "no-version",
    ],
)
def test_a_job_is_read_only_at_a_known_version_and_for_a_job_trigger(
    overrides: dict[str, Any],
) -> None:
    """A resume not implemented here is polled, never guessed at."""
    meta = annotation()
    meta[UIPATH_TASK_META_KEY].update(overrides)

    assert task_job(meta) is None


async def test_a_task_that_names_another_kind_of_trigger_is_polled() -> None:
    """What a server may send later for a resume this client lacks: the run
    stays awake and polls, as for a task with no annotation at all."""
    meta = annotation()
    meta[UIPATH_TASK_META_KEY]["resumeTrigger"] = {"triggerType": "Api"}
    call = ScriptedCall(meta)

    final = await flow_graph(call, SUSPEND_AT_ONCE, MemorySaver()).ainvoke(
        {"result": None}, thread("another-trigger")
    )

    assert final["result"] == "the task's result"
    assert "__interrupt__" not in final
    assert call.waits == [(None, False)]


def test_a_job_named_without_its_folder_is_not_suspended_on() -> None:
    """The resume reads the job in that folder; without it, it would look in
    the run's own folder and fault the run."""
    assert task_job({UIPATH_TASK_META_KEY: {"jobKey": JOB_KEY}}) is None


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {},
        "not-an-object",
        [UIPATH_TASK_META_KEY],
        {UIPATH_TASK_META_KEY: "job"},
        {UIPATH_TASK_META_KEY: {}},
        {UIPATH_TASK_META_KEY: {"jobKey": "not-a-guid"}},
        {UIPATH_TASK_META_KEY: {"jobKey": 42}},
        {UIPATH_TASK_META_KEY: {"jobKey": JOB_KEY, "folderKey": "../elsewhere"}},
    ],
    ids=[
        "no-meta",
        "no-annotation",
        "meta-a-string",
        "meta-a-list",
        "annotation-not-an-object",
        "no-job-key",
        "job-key-not-a-guid",
        "job-key-not-a-string",
        "folder-key-not-a-guid",
    ],
)
def test_an_annotation_that_is_not_exactly_a_job_yields_none(meta: Any) -> None:
    """These values reach a resume trigger; anything odd costs the suspend only."""
    assert task_job(meta) is None


def test_the_callers_polling_time_comes_from_its_flag() -> None:
    # Unset: left to the server, and to the default.
    assert suspension_settings() == SuspensionSettings(None)

    FeatureFlags.configure_flags({MIN_POLLING_TIME_FLAG: "300"})
    assert suspension_settings() == SuspensionSettings(300.0)

    # Zero is a real setting: suspend without polling first.
    FeatureFlags.configure_flags({MIN_POLLING_TIME_FLAG: 0})
    assert suspension_settings() == SuspensionSettings(0.0)


@pytest.mark.parametrize("value", ["soon", -5, "-1", "nan", "inf"])
def test_a_polling_time_that_is_not_a_duration_counts_as_unset(value: Any) -> None:
    FeatureFlags.configure_flags({MIN_POLLING_TIME_FLAG: value})

    assert suspension_settings().min_polling_time_before_suspension is None


def job_suggesting(suspend_after_ms: Any) -> TaskJob:
    meta = annotation()
    meta[UIPATH_TASK_META_KEY]["suspendAfterMs"] = suspend_after_ms
    job = task_job(meta)
    assert job is not None
    return job


def test_the_servers_polling_time_is_read_off_the_annotation() -> None:
    assert job_suggesting(45_000).suspend_after == 45.0
    assert job_suggesting(0).suspend_after == 0.0
    assert task_job(annotation()) == TaskJob(JOB_KEY, FOLDER_KEY, None)


@pytest.mark.parametrize(
    "value",
    [-1, "45000", 1.5, 0.5, True, None, [45000], float("nan"), float("inf")],
)
def test_a_polling_time_the_server_sent_wrong_is_dropped_and_the_job_kept(
    value: Any,
) -> None:
    """It is a hint: a bad one must not cost the suspend."""
    job = job_suggesting(value)

    assert (job.job_key, job.folder_key, job.suspend_after) == (
        JOB_KEY,
        FOLDER_KEY,
        None,
    )


def test_a_polling_time_too_large_for_a_float_is_bounded_like_any_other() -> None:
    """A JSON integer has no upper bound; reading one must not raise."""
    huge = job_suggesting(10**309)

    assert polling_time_before_suspension(SuspensionSettings(), huge) == 840.0


def test_the_polling_time_is_the_callers_then_the_servers_then_the_default() -> None:
    unset = SuspensionSettings()
    silent, suggesting = task_job(annotation()), job_suggesting(45_000)
    assert silent is not None

    assert polling_time_before_suspension(unset, silent) == 60.0
    assert polling_time_before_suspension(unset, suggesting) == 45.0
    assert polling_time_before_suspension(unset, job_suggesting(0)) == 0.0
    # The caller's own setting wins, zero included.
    assert polling_time_before_suspension(SuspensionSettings(5.0), suggesting) == 5.0
    assert polling_time_before_suspension(SuspensionSettings(0.0), suggesting) == 0.0


def test_the_servers_polling_time_is_bounded_by_the_poll_budget() -> None:
    """Nothing the server sends keeps a run awake longer than plain polling would."""
    day = job_suggesting(86_400_000)

    assert polling_time_before_suspension(SuspensionSettings(), day) == 840.0

    FeatureFlags.configure_flags({"McpTasksPollBudgetSeconds": 120})
    assert polling_time_before_suspension(SuspensionSettings(), day) == 120.0
    # The caller's own setting is its own business.
    assert polling_time_before_suspension(SuspensionSettings(900.0), day) == 900.0


def test_the_kill_switch_stops_the_extension_being_declared() -> None:
    from uipath_langchain.agent.tools.mcp.mcp_tasks import session_extension_kwargs

    declared = session_extension_kwargs()["extensions"]
    # With the trigger kinds a run can suspend on, so that the server names a
    # job only to a client that can use one.
    assert declared[UIPATH_TASKS_EXTENSION_ID] == {"resumeTriggerTypes": ["Job"]}

    FeatureFlags.configure_flags({SUSPENSION_KILL_SWITCH: True})

    declared = session_extension_kwargs()["extensions"]
    assert UIPATH_TASKS_EXTENSION_ID not in declared
    # The standard extension is untouched: tasks are still polled.
    assert "io.modelcontextprotocol/tasks" in declared


# --- the flow, in a real graph ---------------------------------------------------


class ScriptedCall:
    """A tool call whose server side is a script.

    Args:
        meta: The ``_meta`` the task handle carries, or ``"no-task"`` for a
            call the server answers at once.
        finishes_in_window: The bounded wait finds the task finished.
        finished_at_park: The one look before suspending finds it finished.
        submit_takes: Seconds the server takes to answer ``tools/call``.
    """

    def __init__(
        self,
        meta: Any = None,
        *,
        finishes_in_window: bool = False,
        finished_at_park: bool = False,
        submit_takes: float = 0,
    ) -> None:
        self.meta = meta
        self.finishes_in_window = finishes_in_window
        self.finished_at_park = finished_at_park
        self.submit_takes = submit_takes
        self.submits = 0
        self.waits: list[tuple[float | None, bool]] = []
        self.peeks = 0

    async def submit(self) -> Submission[str, dict[str, Any]]:
        self.submits += 1
        if self.submit_takes:
            await anyio.sleep(self.submit_takes)
        if self.meta == "no-task":
            return Submission(result="answered at once")
        return Submission(handle={"taskId": "task-1", "_meta": self.meta})

    async def wait(
        self, handle: dict[str, Any], *, seconds: float | None, poll_first: bool
    ) -> str:
        assert handle["taskId"] == "task-1"
        self.waits.append((seconds, poll_first))
        if seconds is not None and not self.finishes_in_window:
            raise PollWindowElapsed()
        return "the task's result"

    async def peek(self, handle: dict[str, Any]) -> str | None:
        self.peeks += 1
        return "finished just now" if self.finished_at_park else None

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


def flow_graph(
    call: ScriptedCall,
    settings: SuspensionSettings = POLL_FIRST,
    checkpointer: MemorySaver | None = None,
    save_results: bool = True,
) -> Any:
    async def node(state: _State) -> dict[str, Any]:
        result = await call_with_suspension(
            call, tool_name="add", settings=settings, save_results=save_results
        )
        return {"result": result}

    builder = StateGraph(_State)
    builder.add_node("call", node)
    builder.add_edge(START, "call")
    builder.add_edge("call", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())


def parked_on(result: dict[str, Any]) -> tuple[str, WaitJobRaw]:
    """The one interrupt a suspended run carries, and the job it waits on."""
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    value = interrupts[0].value
    assert isinstance(value, WaitJobRaw)
    return interrupts[0].id, value


def job_ended(interrupt_id: str) -> Command[Any]:
    """Resume as the runtime does when the Job trigger fires: by interrupt id."""
    ended = Job(key=JOB_KEY, id=59019457, state="successful")
    return Command(resume={interrupt_id: ended})


def thread(name: str) -> Any:
    return {"configurable": {"thread_id": name}}


async def test_a_call_answered_at_once_is_returned_and_nothing_waits() -> None:
    call = ScriptedCall("no-task")

    final = await flow_graph(call).ainvoke({"result": None}, thread("t1"))

    assert final["result"] == "answered at once"
    assert "__interrupt__" not in final
    assert (call.submits, call.waits, call.peeks) == (1, [], 0)


async def test_a_task_with_no_job_named_is_polled_like_any_task() -> None:
    """A server without com.uipath/tasks: nothing to suspend on."""
    call = ScriptedCall(meta=None)

    final = await flow_graph(call).ainvoke({"result": None}, thread("t2"))

    assert final["result"] == "the task's result"
    assert "__interrupt__" not in final
    # One unbounded wait -- the ordinary poll budget -- and no look before it.
    assert (call.waits, call.peeks) == ([(None, False)], 0)


async def test_a_task_is_polled_for_the_minimum_then_suspends_on_its_job() -> None:
    call = ScriptedCall(annotation())
    graph = flow_graph(call)

    first = await graph.ainvoke({"result": None}, thread("t3"))

    interrupt_id, wait = parked_on(first)
    assert wait.job.key == JOB_KEY
    assert wait.process_folder_key == FOLDER_KEY
    # It did not suspend at once: it polled for the minimum first.
    assert len(call.waits) == 1
    window, poll_first = call.waits[0]
    assert window is not None and POLLING_TIME / 2 < window <= POLLING_TIME
    assert poll_first is False
    # And looked once more when the time was up, before giving the run up.
    assert call.peeks == 1

    final = await graph.ainvoke(job_ended(interrupt_id), thread("t3"))

    assert final["result"] == "the task's result"
    # Sent once across both runs, and the resumed run asks for the task at once.
    assert call.submits == 1
    assert call.waits[1:] == [(None, True)]


async def test_a_resumed_run_does_not_poll_the_window_again() -> None:
    """Even when the resume comes at once, as an in-process debug resume does."""
    call = ScriptedCall(annotation())
    graph = flow_graph(call)

    first = await graph.ainvoke({"result": None}, thread("t4"))
    interrupt_id, _ = parked_on(first)
    # Resumed immediately: by the clock, the polling window is still open.
    final = await graph.ainvoke(job_ended(interrupt_id), thread("t4"))

    assert final["result"] == "the task's result"
    assert [seconds is None for seconds, _ in call.waits] == [False, True]


async def test_a_task_that_finishes_within_the_minimum_never_suspends() -> None:
    call = ScriptedCall(annotation(), finishes_in_window=True)

    final = await flow_graph(call).ainvoke({"result": None}, thread("t5"))

    assert final["result"] == "the task's result"
    assert "__interrupt__" not in final
    assert call.peeks == 0


async def test_a_poller_that_stops_early_does_not_cut_the_polling_time_short() -> None:
    """A poll interval longer than the polling time: the poller cannot fit one
    poll and gives up at once. That is not the polling time having passed."""
    call = ScriptedCall(annotation())  # its bounded wait gives up immediately
    started = time.monotonic()

    first = await flow_graph(call).ainvoke({"result": None}, thread("t5a"))

    parked_on(first)
    assert time.monotonic() - started >= POLLING_TIME * 0.9
    assert call.peeks == 1


async def test_a_task_finished_by_the_end_of_the_polling_time_never_suspends() -> None:
    """Found by the last look, after a poller that could not fit a poll."""
    call = ScriptedCall(annotation(), finished_at_park=True)

    final = await flow_graph(call).ainvoke({"result": None}, thread("t5b"))

    assert final["result"] == "finished just now"
    assert "__interrupt__" not in final


async def test_with_no_polling_time_the_run_suspends_after_one_look() -> None:
    call = ScriptedCall(annotation())
    graph = flow_graph(call, SUSPEND_AT_ONCE)

    first = await graph.ainvoke({"result": None}, thread("t6"))

    interrupt_id, _ = parked_on(first)
    assert call.waits == []
    assert call.peeks == 1

    final = await graph.ainvoke(job_ended(interrupt_id), thread("t6"))

    assert final["result"] == "the task's result"
    # That look is part of the replayed step: the resumed run does not repeat it.
    assert (call.submits, call.waits, call.peeks) == (1, [(None, True)], 1)


async def test_a_task_already_finished_at_that_look_costs_a_poll_not_a_suspend() -> (
    None
):
    call = ScriptedCall(annotation(), finished_at_park=True)

    final = await flow_graph(call, SUSPEND_AT_ONCE).ainvoke(
        {"result": None}, thread("t7")
    )

    assert final["result"] == "finished just now"
    assert "__interrupt__" not in final
    assert call.waits == []


def two_call_graph(first: ScriptedCall, second: ScriptedCall) -> Any:
    """A coded node that makes two calls, one after the other."""

    async def node(state: _State) -> dict[str, Any]:
        one = await call_with_suspension(first, tool_name="a", settings=POLL_FIRST)
        two = await call_with_suspension(
            second, tool_name="b", settings=SUSPEND_AT_ONCE
        )
        return {"result": [one, two]}

    builder = StateGraph(_State)
    builder.add_node("call", node)
    builder.add_edge(START, "call")
    builder.add_edge("call", END)
    return builder.compile(checkpointer=MemorySaver())


@pytest.mark.parametrize(
    "first",
    [
        ScriptedCall("no-task"),
        ScriptedCall(None),
        ScriptedCall(annotation(), finishes_in_window=True),
    ],
    ids=["answered-at-once", "polled-task", "task-finished-while-polled"],
)
async def test_an_earlier_call_in_the_node_is_not_sent_again_by_a_later_suspend(
    first: ScriptedCall,
) -> None:
    """The node replays from its first line; the first call must not."""
    first.submits = 0
    second = ScriptedCall(annotation())
    graph = two_call_graph(first, second)

    parked = await graph.ainvoke({"result": None}, thread(f"two-{id(first)}"))
    interrupt_id, _ = parked_on(parked)
    waits_before_the_suspend = list(first.waits)
    final = await graph.ainvoke(job_ended(interrupt_id), thread(f"two-{id(first)}"))

    assert final["result"][1] == "the task's result"
    assert final["result"][0] in ("answered at once", "the task's result")
    assert (first.submits, second.submits) == (1, 1)
    # Nor is its task asked for again: by the replay the server may have let
    # the task go, and the result is the one that was saved.
    assert first.waits == waits_before_the_suspend


async def test_the_servers_polling_time_decides_when_the_run_suspends() -> None:
    """No setting on the caller's side: a server that says zero is suspended on
    at once, and one that says a minute is polled first."""
    at_once = annotation()
    at_once[UIPATH_TASK_META_KEY]["suspendAfterMs"] = 0
    call = ScriptedCall(at_once)
    parked = await flow_graph(call, SuspensionSettings(), MemorySaver()).ainvoke(
        {"result": None}, thread("server-says-zero")
    )
    parked_on(parked)
    assert call.waits == []  # no polling before the suspend

    a_minute = annotation()
    a_minute[UIPATH_TASK_META_KEY]["suspendAfterMs"] = 60_000
    call = ScriptedCall(a_minute, finishes_in_window=True)
    final = await flow_graph(call, SuspensionSettings(), MemorySaver()).ainvoke(
        {"result": None}, thread("server-says-a-minute")
    )
    assert final["result"] == "the task's result"
    [(seconds, _)] = call.waits
    assert seconds is not None and 55 < seconds <= 60


async def test_the_time_the_server_takes_to_answer_counts_as_polling_time() -> None:
    """The polling time runs from when the call was sent, not from its answer."""
    call = ScriptedCall(annotation(), finishes_in_window=True, submit_takes=0.5)

    await flow_graph(call, SuspensionSettings(60.0), MemorySaver()).ainvoke(
        {"result": None}, thread("slow-to-answer")
    )

    [(seconds, _)] = call.waits
    assert seconds is not None and seconds <= 59.6


async def test_a_call_that_suspended_and_resumed_is_not_polled_again_by_a_later_suspend() -> (
    None
):
    """Two calls in one node, both long: after the first one's resume its
    result is saved, so the second one's resume does not read its task again."""
    first, second = ScriptedCall(annotation()), ScriptedCall(annotation())

    async def node(state: _State) -> dict[str, Any]:
        one = await call_with_suspension(first, tool_name="a", settings=SUSPEND_AT_ONCE)
        two = await call_with_suspension(
            second, tool_name="b", settings=SUSPEND_AT_ONCE
        )
        return {"result": [one, two]}

    builder = StateGraph(_State)
    builder.add_node("call", node)
    builder.add_edge(START, "call")
    builder.add_edge("call", END)
    graph = builder.compile(checkpointer=MemorySaver())

    parked = await graph.ainvoke({"result": None}, thread("two-long"))
    first_interrupt, _ = parked_on(parked)
    parked = await graph.ainvoke(job_ended(first_interrupt), thread("two-long"))
    second_interrupt, _ = parked_on(parked)
    assert first.waits == [(None, True)]

    final = await graph.ainvoke(job_ended(second_interrupt), thread("two-long"))

    assert final["result"] == ["the task's result", "the task's result"]
    assert (first.submits, second.submits) == (1, 1)
    assert first.waits == [(None, True)]
    assert second.waits == [(None, True)]


def resume_writes(checkpointer: MemorySaver, name: str) -> str:
    """Everything the steps of a run saved in its checkpoint, as text."""
    writes = [
        value
        for checkpoint in checkpointer.list(thread(name))
        for _, channel, value in checkpoint.pending_writes or []
        if channel == "__resume__"
    ]
    return json.dumps(writes, default=str)


async def test_a_tool_node_keeps_results_out_of_the_checkpoint() -> None:
    """``save_results=False``: one call per node and nothing after it, so the
    result -- which can be large -- is not stored a second time."""
    for name, call in (
        ("at-once", ScriptedCall("no-task")),
        ("polled", ScriptedCall(None)),
        ("in-window", ScriptedCall(annotation(), finishes_in_window=True)),
    ):
        checkpointer = MemorySaver()
        graph = flow_graph(call, checkpointer=checkpointer, save_results=False)

        final = await graph.ainvoke({"result": None}, thread(name))

        assert final["result"] in ("answered at once", "the task's result")
        saved = resume_writes(checkpointer, name)
        assert "answered at once" not in saved and "the task's result" not in saved

    # Saved by default, which is what a coded node relies on.
    checkpointer = MemorySaver()
    await flow_graph(ScriptedCall("no-task"), checkpointer=checkpointer).ainvoke(
        {"result": None}, thread("saved")
    )
    assert "answered at once" in resume_writes(checkpointer, "saved")


async def test_a_tool_node_suspends_and_resumes_without_saving_the_result() -> None:
    call = ScriptedCall(annotation())
    checkpointer = MemorySaver()
    graph = flow_graph(call, SUSPEND_AT_ONCE, checkpointer, save_results=False)

    first = await graph.ainvoke({"result": None}, thread("unsaved"))
    interrupt_id, _ = parked_on(first)
    final = await graph.ainvoke(job_ended(interrupt_id), thread("unsaved"))

    assert final["result"] == "the task's result"
    assert call.submits == 1
    assert "the task's result" not in resume_writes(checkpointer, "unsaved")


async def test_the_kill_switch_thrown_during_a_suspend_does_not_send_the_call_again() -> (
    None
):
    """The replay stays on the saved steps and reads the task it already has."""
    call = ScriptedCall(annotation())

    async def node(state: _State) -> dict[str, Any]:
        # As the callers do: the plain path, unless a durable call is in play.
        if not suspension.durable_call_in_play():
            await call.submit()
            return {"result": "sent again"}
        result = await call_with_suspension(
            call, tool_name="add", settings=SUSPEND_AT_ONCE
        )
        return {"result": result}

    builder = StateGraph(_State)
    builder.add_node("call", node)
    builder.add_edge(START, "call")
    builder.add_edge("call", END)
    graph = builder.compile(checkpointer=MemorySaver())

    first = await graph.ainvoke({"result": None}, thread("switch"))
    interrupt_id, _ = parked_on(first)
    FeatureFlags.configure_flags({SUSPENSION_KILL_SWITCH: True})
    final = await graph.ainvoke(job_ended(interrupt_id), thread("switch"))

    assert final["result"] == "the task's result"
    assert call.submits == 1


async def test_with_the_kill_switch_on_a_durable_call_polls_and_never_suspends() -> (
    None
):
    """Reached on a replay of a node that suspended on something else."""
    FeatureFlags.configure_flags({SUSPENSION_KILL_SWITCH: True})
    call = ScriptedCall(annotation())

    final = await flow_graph(call, SUSPEND_AT_ONCE).ainvoke(
        {"result": None}, thread("switch-on")
    )

    assert final["result"] == "the task's result"
    assert "__interrupt__" not in final
    assert (call.waits, call.peeks) == ([(None, False)], 0)


async def test_a_call_made_after_an_interrupt_in_the_same_node_fails_and_is_not_resent() -> (
    None
):
    """The documented limit: the steps are counted from the start of the node,
    and LangGraph's own interrupt is not one of them. Such a call cannot be
    resumed -- and what it must never do instead is send the call."""
    call = ScriptedCall("no-task")

    async def node(state: _State) -> dict[str, Any]:
        interrupt("approve?")
        result = await call_with_suspension(call, tool_name="add")
        return {"result": result}

    builder = StateGraph(_State)
    builder.add_node("call", node)
    builder.add_edge(START, "call")
    builder.add_edge("call", END)
    graph = builder.compile(checkpointer=MemorySaver())

    await graph.ainvoke({"result": None}, thread("native"))
    stalled = await graph.ainvoke(Command(resume="yes"), thread("native"))
    assert "__interrupt__" in stalled

    with pytest.raises(AgentRuntimeError, match="could not be read back"):
        await graph.ainvoke(Command(resume="again"), thread("native"))

    assert call.submits == 0


async def test_what_is_saved_to_suspend_is_enough_to_poll_again() -> None:
    """The checkpoint holds the handle: the task id, the job, the send time."""
    call = ScriptedCall(annotation())
    checkpointer = MemorySaver()
    graph = flow_graph(call, SUSPEND_AT_ONCE, checkpointer)

    await graph.ainvoke({"result": None}, thread("t8"))

    writes = [
        value
        for checkpoint in checkpointer.list(thread("t8"))
        for _, channel, value in checkpoint.pending_writes or []
        if channel == "__resume__"
    ]
    saved = json.dumps(writes, default=str)
    assert "task-1" in saved and JOB_KEY in saved
    assert "submitted_at" in saved


# --- the MCP tool in its own node, over real HTTP --------------------------------


def declares_uipath_tasks(body: dict[str, Any]) -> bool:
    """Whether a request declares ``com.uipath/tasks`` and can suspend on a job.

    As AgentHub reads it: a declaration that lists no trigger kinds means Job.
    """
    meta = (body.get("params") or {}).get("_meta") or {}
    capabilities = meta.get("io.modelcontextprotocol/clientCapabilities") or {}
    extensions = capabilities.get("extensions") or {}
    if UIPATH_TASKS_EXTENSION_ID not in extensions:
        return False
    declared = extensions[UIPATH_TASKS_EXTENSION_ID]
    if not isinstance(declared, dict) or "resumeTriggerTypes" not in declared:
        return True
    kinds = declared["resumeTriggerTypes"]
    return isinstance(kinds, list) and "Job" in kinds


class JobBackedTaskServer(TaskServingServer):
    """A task server that serves ``com.uipath/tasks``: its handle names a job.

    As AgentHub does, it names the job only to a caller that declared the
    extension on the request.

    Args:
        names_the_job_to_anyone: Name the job whatever the caller declared.
            For the coded-agent tests of the flow itself: the platform SDK's
            ``McpService`` declares the extension only from 0.3.19.
    """

    def __init__(self, *, names_the_job_to_anyone: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.names_the_job_to_anyone = names_the_job_to_anyone

    def _call_tool(self, body: dict[str, Any]) -> Any:
        response = super()._call_tool(body)
        if not (self.names_the_job_to_anyone or declares_uipath_tasks(body)):
            return response
        envelope = json.loads(bytes(response.body))
        envelope["result"]["_meta"] = annotation()
        return self._result(body["id"], envelope["result"])


#: The task never finishes on its own: only a job ending (the test) finishes it.
NEVER = 10_000


def polling_time(seconds: float) -> None:
    FeatureFlags.configure_flags({MIN_POLLING_TIME_FLAG: seconds})


async def tool_graph(checkpointer: MemorySaver) -> tuple[Any, Any, Any]:
    """A fresh client, its tool, and a one-node graph around the tool's node."""
    config = make_resource_config()
    client = make_client(protocol_mode="modern")
    tools = await create_mcp_tools(config, client)
    tool = tools[0]

    builder = StateGraph(AgentGraphState)
    builder.add_node("tool", create_tool_node(tools)[tool.name])
    builder.add_edge(START, "tool")
    builder.add_edge("tool", END)
    return builder.compile(checkpointer=checkpointer), tool, client


def tool_call(tool: Any) -> AgentGraphState:
    call = {"name": tool.name, "args": {"a": 2, "b": 3}, "id": "call-1"}
    return AgentGraphState(messages=[AIMessage(content="", tool_calls=[call])])


async def test_the_tool_suspends_and_a_new_process_reads_the_task() -> None:
    """The whole feature: send once, suspend on the job, resume elsewhere."""
    polling_time(0)
    endpoint = JobBackedTaskServer(polls_before_done=NEVER)
    gateway = RecordingGateway(endpoint.build_app())
    checkpointer = MemorySaver()

    async with serve(gateway) as url:
        with patched_sdk(url):
            graph, tool, client = await tool_graph(checkpointer)
            try:
                first = await graph.ainvoke(tool_call(tool), thread("run"))
            finally:
                await client.dispose()

            interrupt_id, wait = parked_on(first)
            assert wait.job.key == JOB_KEY
            assert wait.process_folder_key == FOLDER_KEY

            # The job ends; Orchestrator starts the run again. Nothing of the
            # first process is left but the checkpoint.
            endpoint.polls_before_done = 0
            graph, tool, client = await tool_graph(checkpointer)
            try:
                final = await graph.ainvoke(job_ended(interrupt_id), thread("run"))
            finally:
                await client.dispose()

    assert "__interrupt__" not in final
    assert "5" in str(final["messages"][-1].content)
    # One job for one tool call, across two processes.
    assert endpoint.task_calls == 1
    assert gateway.count("tools/call") == 1

    declared = gateway.for_rpc("tools/call")[0].meta[
        "io.modelcontextprotocol/clientCapabilities"
    ]["extensions"]
    assert declared[UIPATH_TASKS_EXTENSION_ID] == {"resumeTriggerTypes": ["Job"]}
    assert "io.modelcontextprotocol/tasks" in declared


async def test_the_tool_polls_a_task_before_it_suspends() -> None:
    polling_time(0.6)
    endpoint = JobBackedTaskServer(polls_before_done=NEVER, poll_interval_ms=50)
    gateway = RecordingGateway(endpoint.build_app())
    checkpointer = MemorySaver()

    async with serve(gateway) as url:
        with patched_sdk(url):
            graph, tool, client = await tool_graph(checkpointer)
            try:
                first = await graph.ainvoke(tool_call(tool), thread("run"))
                interrupt_id, _ = parked_on(first)
                polls_before_suspending = endpoint.poll_count

                endpoint.polls_before_done = 0
                final = await graph.ainvoke(job_ended(interrupt_id), thread("run"))
            finally:
                await client.dispose()

    # It polled through the window before suspending (the client polls no
    # faster than every 100 ms), and the resumed run needed exactly one more.
    assert polls_before_suspending >= 2
    assert endpoint.poll_count == polls_before_suspending + 1
    assert "5" in str(final["messages"][-1].content)
    assert endpoint.task_calls == 1


async def test_the_tool_waits_out_the_polling_time_when_no_poll_fits_in_it() -> None:
    """A server asking for polls further apart than the polling time."""
    polling_time(0.4)
    endpoint = JobBackedTaskServer(polls_before_done=NEVER, poll_interval_ms=60_000)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        with patched_sdk(url):
            graph, tool, client = await tool_graph(MemorySaver())
            started = time.monotonic()
            try:
                first = await graph.ainvoke(tool_call(tool), thread("run"))
            finally:
                await client.dispose()

    parked_on(first)
    # Not at once: the time was waited out, and the task looked at once.
    assert time.monotonic() - started >= 0.35
    assert endpoint.poll_count == 1


async def test_the_kill_switch_thrown_while_the_tool_is_suspended_starts_no_second_job() -> (
    None
):
    polling_time(0)
    endpoint = JobBackedTaskServer(polls_before_done=NEVER)
    gateway = RecordingGateway(endpoint.build_app())
    checkpointer = MemorySaver()

    async with serve(gateway) as url:
        with patched_sdk(url):
            graph, tool, client = await tool_graph(checkpointer)
            try:
                first = await graph.ainvoke(tool_call(tool), thread("run"))
            finally:
                await client.dispose()
            interrupt_id, _ = parked_on(first)

            FeatureFlags.configure_flags({SUSPENSION_KILL_SWITCH: True})
            endpoint.polls_before_done = 0
            graph, tool, client = await tool_graph(checkpointer)
            try:
                final = await graph.ainvoke(job_ended(interrupt_id), thread("run"))
            finally:
                await client.dispose()

    assert "5" in str(final["messages"][-1].content)
    assert endpoint.task_calls == 1
    assert gateway.count("tools/call") == 1


class FirstPollFails(JobBackedTaskServer):
    """Answers the first ``tasks/get`` after ``fail_from`` with a gateway's 503."""

    fail_poll: int | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.poll_times: list[float] = []

    def _tasks_get(self, body: dict[str, Any]) -> Any:
        self.poll_times.append(time.monotonic())
        if self.fail_poll is not None and len(self.poll_times) == self.fail_poll:
            return Response("bad gateway", status_code=503)
        return super()._tasks_get(body)


async def test_a_failed_first_poll_after_the_resume_is_retried_on_the_interval() -> (
    None
):
    """Only the first request goes out at once; its retry is not a tight loop."""
    polling_time(0)
    endpoint = FirstPollFails(polls_before_done=NEVER, poll_interval_ms=300)
    gateway = RecordingGateway(endpoint.build_app())
    checkpointer = MemorySaver()

    async with serve(gateway) as url:
        with patched_sdk(url):
            graph, tool, client = await tool_graph(checkpointer)
            try:
                first = await graph.ainvoke(tool_call(tool), thread("run"))
            finally:
                await client.dispose()
            interrupt_id, _ = parked_on(first)

            endpoint.polls_before_done = 0
            endpoint.fail_poll = len(endpoint.poll_times) + 1
            graph, tool, client = await tool_graph(checkpointer)
            try:
                final = await graph.ainvoke(job_ended(interrupt_id), thread("run"))
            finally:
                await client.dispose()

    assert "5" in str(final["messages"][-1].content)
    failed, retried = endpoint.poll_times[-2:]
    assert retried - failed >= 0.25


async def test_a_look_that_does_not_get_through_does_not_stop_the_suspend() -> None:
    """A gateway's 503 on the look before suspending says nothing of the task."""
    polling_time(0)
    endpoint = FirstPollFails(polls_before_done=NEVER)
    endpoint.fail_poll = 1
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        with patched_sdk(url):
            graph, tool, client = await tool_graph(MemorySaver())
            try:
                first = await graph.ainvoke(tool_call(tool), thread("run"))
            finally:
                await client.dispose()

    parked_on(first)
    assert len(endpoint.poll_times) == 1


def test_a_tool_result_saved_with_its_step_comes_back_as_it_was() -> None:
    """Saved in the checkpoint as JSON, wire names and non-text blocks included."""
    from mcp.types import CallToolResult, ImageContent, TextContent

    from uipath_langchain.agent.tools.mcp.mcp_tool import _SessionTaskCall

    call = _SessionTaskCall(None, None, {}, False, None)  # type: ignore[arg-type]
    result = CallToolResult(
        content=[
            TextContent(type="text", text="5"),
            ImageContent(type="image", data="aGk=", mime_type="image/png"),
        ],
        structured_content={"result": 5},
    )

    saved = json.loads(json.dumps(call.dump_result(result)))

    assert saved["result"]["content"][1]["mimeType"] == "image/png"
    assert call.load_result(saved) == result
    # The schema check's "retry" answer is a string, and stays one.
    assert call.load_result(call.dump_result("retry with x")) == "retry with x"


async def test_the_tool_never_suspends_a_task_that_finishes_while_it_polls() -> None:
    """The default minute of polling: a task done within it costs no suspend."""
    endpoint = JobBackedTaskServer(polls_before_done=2)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        with patched_sdk(url):
            graph, tool, client = await tool_graph(MemorySaver())
            try:
                final = await graph.ainvoke(tool_call(tool), thread("run"))
            finally:
                await client.dispose()

    assert "__interrupt__" not in final
    assert "5" in str(final["messages"][-1].content)
    assert endpoint.poll_count == 3


async def test_the_kill_switch_keeps_the_tool_polling() -> None:
    FeatureFlags.configure_flags(
        {SUSPENSION_KILL_SWITCH: True, MIN_POLLING_TIME_FLAG: 0}
    )
    endpoint = JobBackedTaskServer(polls_before_done=2)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        with patched_sdk(url):
            graph, tool, client = await tool_graph(MemorySaver())
            try:
                final = await graph.ainvoke(tool_call(tool), thread("run"))
            finally:
                await client.dispose()

    assert "__interrupt__" not in final
    assert "5" in str(final["messages"][-1].content)


async def test_the_tool_reached_from_another_node_polls_instead_of_suspending() -> None:
    """A bridge runs the tool inside someone else's node; that node must not replay."""
    polling_time(0)
    endpoint = JobBackedTaskServer(polls_before_done=2)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        with patched_sdk(url):
            client = make_client(protocol_mode="modern")
            tools = await create_mcp_tools(make_resource_config(), client)

            async def bridge(state: _State) -> dict[str, Any]:
                return {"result": await tools[0].ainvoke({"a": 2, "b": 3})}

            builder = StateGraph(_State)
            builder.add_node("bridge", bridge)
            builder.add_edge(START, "bridge")
            builder.add_edge("bridge", END)
            graph = builder.compile(checkpointer=MemorySaver())
            try:
                final = await graph.ainvoke({"result": None}, thread("run"))
            finally:
                await client.dispose()

    assert "__interrupt__" not in final
    assert "5" in str(final["result"])
    assert endpoint.poll_count == 3


def test_the_scripted_call_is_the_protocol_the_flow_drives() -> None:
    """Keeps the fake honest: it must satisfy the flow's own protocol."""
    call: suspension.TaskCall[str, dict[str, Any]] = ScriptedCall()
    assert call.job({"_meta": annotation()}) == TaskJob(JOB_KEY, FOLDER_KEY)


# --- a coded agent: call_mcp_tool over the platform SDK's McpService ---------------


def coded_sdk(url: str) -> Any:
    """A real ``McpService`` whose server lookup resolves to a local endpoint."""
    from uipath.platform import UiPathApiConfig, UiPathExecutionContext
    from uipath.platform.orchestrator import McpService
    from uipath.platform.orchestrator._folder_service import FolderService
    from uipath.platform.orchestrator.mcp import McpServer

    config = UiPathApiConfig(base_url="https://test.uipath.com/org/tenant", secret="s")
    context = UiPathExecutionContext()
    service = McpService(
        config=config,
        execution_context=context,
        folders_service=FolderService(config=config, execution_context=context),
    )

    async def retrieve_async(slug: str, folder_path: str | None = None) -> McpServer:
        return McpServer(
            id="mcp-server-id",
            name=slug,
            slug=slug,
            folderKey=FOLDER_KEY,
            mcpUrl=url,
        )

    service.retrieve_async = retrieve_async

    class _Sdk:
        mcp = service

    return _Sdk()


def coded_graph(checkpointer: MemorySaver) -> Any:
    from uipath_langchain.agent.tools.mcp import call_mcp_tool

    async def node(state: _State) -> dict[str, Any]:
        result = await call_mcp_tool("math", PINNED_TOOL_NAME, {"a": 2, "b": 3})
        return {"result": result}

    builder = StateGraph(_State)
    builder.add_node("call", node)
    builder.add_edge(START, "call")
    builder.add_edge("call", END)
    return builder.compile(checkpointer=checkpointer)


async def test_a_coded_agent_suspends_on_the_job_and_reads_the_task_on_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uipath_langchain.agent.tools.mcp import mcp_service_call

    polling_time(0)
    endpoint = JobBackedTaskServer(
        names_the_job_to_anyone=True, polls_before_done=NEVER
    )
    gateway = RecordingGateway(endpoint.build_app())
    checkpointer = MemorySaver()

    async with serve(gateway) as url:
        monkeypatch.setattr(mcp_service_call, "UiPath", lambda: coded_sdk(url))

        first = await coded_graph(checkpointer).ainvoke(
            {"result": None}, thread("coded")
        )
        interrupt_id, wait = parked_on(first)
        assert wait.job.key == JOB_KEY

        # The job ends and the run starts again: a new graph, a new SDK client.
        endpoint.polls_before_done = 0
        final = await coded_graph(checkpointer).ainvoke(
            job_ended(interrupt_id), thread("coded")
        )

    assert final["result"]["structuredContent"] == {"result": 5}
    assert endpoint.task_calls == 1
    assert gateway.count("tools/call") == 1


class PollFailsInPassing(JobBackedTaskServer):
    """Answers one chosen ``tasks/get`` with a JSON-RPC internal error."""

    def __init__(self, *, fail_poll: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fail_poll = fail_poll
        self.polls = 0

    def _tasks_get(self, body: dict[str, Any]) -> Any:
        self.polls += 1
        if self.polls == self.fail_poll:
            return self._error(body["id"], -32603, "Internal error", status=200)
        return super()._tasks_get(body)


async def test_a_coded_agent_rides_out_a_failed_first_read_after_the_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool's job succeeded; one failed read must not fail the run, and
    the call must not be sent again."""
    from uipath_langchain.agent.tools.mcp import mcp_service_call

    polling_time(0)
    # Poll 1 is the look before suspending; poll 2 is the first after the resume.
    endpoint = PollFailsInPassing(
        fail_poll=2,
        names_the_job_to_anyone=True,
        polls_before_done=NEVER,
        poll_interval_ms=50,
    )
    gateway = RecordingGateway(endpoint.build_app())
    checkpointer = MemorySaver()

    async with serve(gateway) as url:
        monkeypatch.setattr(mcp_service_call, "UiPath", lambda: coded_sdk(url))
        first = await coded_graph(checkpointer).ainvoke(
            {"result": None}, thread("coded-blip")
        )
        interrupt_id, _ = parked_on(first)

        endpoint.polls_before_done = 0
        final = await coded_graph(checkpointer).ainvoke(
            job_ended(interrupt_id), thread("coded-blip")
        )

    assert final["result"]["structuredContent"] == {"result": 5}
    assert endpoint.polls == 3  # the look, the failed read, the read that got it
    assert gateway.count("tools/call") == 1


async def test_a_coded_agent_suspends_when_the_look_before_it_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A look that failed says nothing about the task: the run suspends on the
    job as it would have, instead of failing a call whose job is running."""
    from uipath_langchain.agent.tools.mcp import mcp_service_call

    polling_time(0)
    endpoint = PollFailsInPassing(
        fail_poll=1,
        names_the_job_to_anyone=True,
        polls_before_done=NEVER,
        poll_interval_ms=50,
    )
    gateway = RecordingGateway(endpoint.build_app())
    checkpointer = MemorySaver()

    async with serve(gateway) as url:
        monkeypatch.setattr(mcp_service_call, "UiPath", lambda: coded_sdk(url))
        first = await coded_graph(checkpointer).ainvoke(
            {"result": None}, thread("coded-blind-look")
        )
        interrupt_id, wait = parked_on(first)
        assert wait.job.key == JOB_KEY

        endpoint.polls_before_done = 0
        final = await coded_graph(checkpointer).ainvoke(
            job_ended(interrupt_id), thread("coded-blind-look")
        )

    assert final["result"]["structuredContent"] == {"result": 5}
    assert gateway.count("tools/call") == 1


async def test_a_coded_agent_polls_a_task_that_finishes_in_time_to_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uipath_langchain.agent.tools.mcp import mcp_service_call

    endpoint = JobBackedTaskServer(names_the_job_to_anyone=True, polls_before_done=2)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        monkeypatch.setattr(mcp_service_call, "UiPath", lambda: coded_sdk(url))
        final = await coded_graph(MemorySaver()).ainvoke(
            {"result": None}, thread("coded-short")
        )

    assert "__interrupt__" not in final
    assert final["result"]["structuredContent"] == {"result": 5}


async def test_a_coded_call_already_finished_at_the_look_never_suspends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uipath_langchain.agent.tools.mcp import mcp_service_call

    polling_time(0)
    endpoint = JobBackedTaskServer(names_the_job_to_anyone=True, polls_before_done=0)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        monkeypatch.setattr(mcp_service_call, "UiPath", lambda: coded_sdk(url))
        final = await coded_graph(MemorySaver()).ainvoke(
            {"result": None}, thread("coded-done")
        )

    assert "__interrupt__" not in final
    assert final["result"]["structuredContent"] == {"result": 5}
    assert endpoint.poll_count == 1


class AnsweringAtOnce(JobBackedTaskServer):
    """Serves tasks, but answers this call with an ordinary result."""

    def _call_tool(self, body: dict[str, Any]) -> Any:
        self.task_calls += 1
        return PinnedVersionServer._call_tool(self, body)


async def test_a_coded_call_answered_at_once_is_sent_once_across_a_later_suspend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two calls in one node: the second suspends, and the node replays."""
    from uipath_langchain.agent.tools.mcp import call_mcp_tool, mcp_service_call

    polling_time(0)
    quick = AnsweringAtOnce()
    slow = JobBackedTaskServer(names_the_job_to_anyone=True, polls_before_done=NEVER)
    checkpointer = MemorySaver()

    async def node(state: _State) -> dict[str, Any]:
        monkeypatch.setattr(mcp_service_call, "UiPath", lambda: coded_sdk(quick_url))
        one = await call_mcp_tool("math", PINNED_TOOL_NAME, {"a": 1, "b": 1})
        monkeypatch.setattr(mcp_service_call, "UiPath", lambda: coded_sdk(slow_url))
        two = await call_mcp_tool("math", PINNED_TOOL_NAME, {"a": 2, "b": 3})
        return {"result": [one, two]}

    builder = StateGraph(_State)
    builder.add_node("call", node)
    builder.add_edge(START, "call")
    builder.add_edge("call", END)
    graph = builder.compile(checkpointer=checkpointer)

    async with (
        serve(quick.build_app()) as quick_url,
        serve(slow.build_app()) as slow_url,
    ):
        first = await graph.ainvoke({"result": None}, thread("coded-two"))
        interrupt_id, _ = parked_on(first)

        slow.polls_before_done = 0
        final = await graph.ainvoke(job_ended(interrupt_id), thread("coded-two"))

    assert "2" in json.dumps(final["result"][0])
    assert final["result"][1]["structuredContent"] == {"result": 5}
    assert (quick.task_calls, slow.task_calls) == (1, 1)


async def test_a_coded_agent_suspends_only_where_its_sdk_declares_the_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Against a server that names the job only to a declaring caller, as
    AgentHub does. ``McpService`` declares ``com.uipath/tasks`` from
    uipath-platform 0.3.19; an older one never learns the job, and polls."""
    from uipath.platform.orchestrator import mcp_tasks as platform_tasks

    from uipath_langchain.agent.tools.mcp import mcp_service_call

    sdk_declares = hasattr(platform_tasks, "MCP_UIPATH_TASKS_EXTENSION_ID")
    polling_time(0)
    endpoint = JobBackedTaskServer(polls_before_done=2)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        monkeypatch.setattr(mcp_service_call, "UiPath", lambda: coded_sdk(url))
        outcome = await coded_graph(MemorySaver()).ainvoke(
            {"result": None}, thread("coded-negotiated")
        )

    if sdk_declares:
        parked_on(outcome)
    else:
        assert "__interrupt__" not in outcome
        assert outcome["result"]["structuredContent"] == {"result": 5}
    assert endpoint.task_calls == 1


async def test_outside_a_graph_node_a_coded_call_is_a_polled_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing to suspend: the task is polled, never an interrupt raised."""
    from uipath_langchain.agent.tools.mcp import call_mcp_tool, mcp_service_call

    polling_time(0)
    endpoint = JobBackedTaskServer(names_the_job_to_anyone=True, polls_before_done=1)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        monkeypatch.setattr(mcp_service_call, "UiPath", lambda: coded_sdk(url))
        result = await call_mcp_tool("math", PINNED_TOOL_NAME, {"a": 2, "b": 3})

    assert result["structuredContent"] == {"result": 5}


# --- what Orchestrator is asked to wait on ----------------------------------------


async def test_the_suspend_becomes_a_job_trigger_on_the_tool_job() -> None:
    """The interrupt, through the real trigger handler: a Job trigger by key."""
    from uipath.core.triggers import UiPathResumeTriggerName, UiPathResumeTriggerType
    from uipath.platform.resume_triggers import UiPathResumeTriggerHandler

    wait = suspension._wait_on_job(TaskJob(JOB_KEY, FOLDER_KEY))

    trigger = await UiPathResumeTriggerHandler().create_trigger(wait)

    assert trigger.trigger_type == UiPathResumeTriggerType.JOB
    # Raw: the resume must not read the job's output or fault on a failed job.
    # The result is read with tasks/get.
    assert trigger.trigger_name == UiPathResumeTriggerName.JOB_RAW
    assert trigger.item_key == JOB_KEY
    assert trigger.folder_key == FOLDER_KEY


# --- a low-code agent: the whole agent graph, as uipath-agents builds it -----------


class _Answer(BaseModel):
    answer: str


def scripted_model(*replies: AIMessage) -> Any:
    """A chat model that answers with ``replies`` in turn and records its input."""
    model: Any = _StubChatAnthropic.model_construct(model="claude-opus-5-5")
    model.model_details = {}
    model.bind_tools = Mock(return_value=model)
    model.ainvoke = AsyncMock(side_effect=list(replies))
    return model


async def low_code_agent(model: Any, checkpointer: MemorySaver) -> tuple[Any, Any, Any]:
    """The graph a low-code agent with one MCP resource runs.

    Built the way ``uipath_agents`` builds it: ``create_mcp_tools`` for the
    resource, then ``create_agent`` over the tools -- the LLM node, the tool's
    own node behind the guardrails subgraph, and the terminate node.
    """
    client = make_client(protocol_mode="modern")
    tools = await create_mcp_tools(make_resource_config(), client)
    graph: Any = create_agent(
        model,
        tools,
        [SystemMessage(content="Add the numbers.")],
        output_schema=_Answer,
    ).compile(checkpointer=checkpointer)
    return graph, tools[0], client


async def test_a_low_code_agent_suspends_on_its_mcp_tool_and_finishes_after_the_resume() -> (
    None
):
    """The model calls the MCP tool; the job suspends; a new process finishes."""
    polling_time(0)
    endpoint = JobBackedTaskServer(polls_before_done=NEVER)
    gateway = RecordingGateway(endpoint.build_app())
    checkpointer = MemorySaver()
    done = AIMessage(
        content="",
        tool_calls=[
            create_tool_call(name="end_execution", args={"answer": "5"}, id="c2")
        ],
    )

    async with serve(gateway) as url:
        with patched_sdk(url):
            model = scripted_model(done)  # placeholder until the tool's name is known
            graph, tool, client = await low_code_agent(model, checkpointer)
            model.ainvoke.side_effect = [
                AIMessage(
                    content="",
                    tool_calls=[
                        create_tool_call(name=tool.name, args={"a": 2, "b": 3}, id="c1")
                    ],
                )
            ]
            try:
                first = await graph.ainvoke({}, thread("low-code"))
            finally:
                await client.dispose()

            # The run is suspended on the job behind the MCP task, not polling.
            interrupt_id, wait = parked_on(first)
            assert wait.job.key == JOB_KEY

            # The job ends; Orchestrator starts the agent again from the checkpoint.
            endpoint.polls_before_done = 0
            resumed_model = scripted_model(done)
            graph, tool, client = await low_code_agent(resumed_model, checkpointer)
            try:
                final = await graph.ainvoke(job_ended(interrupt_id), thread("low-code"))
            finally:
                await client.dispose()

    assert final == {"answer": "5"}
    # One job for one tool call, and the model was shown the tool's result.
    assert endpoint.task_calls == 1
    assert gateway.count("tools/call") == 1
    shown = resumed_model.ainvoke.call_args.args[0]
    assert any(
        isinstance(message, ToolMessage) and "5" in str(message.content)
        for message in shown
    )
