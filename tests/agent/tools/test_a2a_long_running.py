"""Long-running A2A calls: send once, then poll on Orchestrator timer wake-ups.

The graph tests run the tool node under a checkpointer and resume it the way the
runtime does after a Timer trigger fires (``{"resumeTime": ...}``), to check the
replay is deterministic: the message is sent once, and each run polls at most once.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, TypedDict

import httpx
import pytest
from a2a.client.errors import A2AClientError, A2AClientTimeoutError
from a2a.types import (
    Artifact,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.errors import (
    A2AError,
    InternalError,
    InvalidRequestError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from google.protobuf.json_format import ParseError
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from uipath.agent.models.agent import (
    AgentA2aResourceConfig,
    AgentConversationalAgentToolProperties,
    AgentConversationalAgentToolResourceConfig,
    AgentToolType,
)
from uipath.platform.common import WaitUntil

import uipath_langchain.agent.tools.a2a.long_running as long_running
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.a2a.a2a_tool import create_a2a_tools_and_clients
from uipath_langchain.agent.tools.a2a.conversational_agent_tool import (
    create_conversational_agent_tools_and_clients,
)
from uipath_langchain.agent.tools.a2a.long_running import (
    A2aPollSettings,
    A2aTaskLostError,
    send_and_wait,
)
from uipath_langchain.agent.tools.tool_node import create_tool_node

START_TIME = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)

NO_INLINE_WAIT = A2aPollSettings(
    initial_inline_wait=timedelta(0),
    poll_intervals=(timedelta(minutes=5), timedelta(minutes=10)),
    max_wait=timedelta(hours=1),
)


class _Clock:
    def __init__(self) -> None:
        self.now = START_TIME

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(long_running, "_utcnow", clock)

    async def _no_sleep(seconds: float) -> None:
        clock.advance(timedelta(seconds=seconds))

    monkeypatch.setattr(long_running, "asyncio", SimpleNamespace(sleep=_no_sleep))
    return clock


def _task(state: TaskState, *, text: str = "", task_id: str = "task-1") -> Task:
    task = Task(id=task_id, context_id="ctx-1", status=TaskStatus(state=state))
    if text:
        task.artifacts.append(Artifact(artifact_id="a-1", parts=[Part(text=text)]))
    return task


def _working() -> Task:
    return _task(TaskState.TASK_STATE_WORKING)


def _completed(text: str = "report ready") -> Task:
    return _task(TaskState.TASK_STATE_COMPLETED, text=text)


class _ScriptedClient:
    """A2A client whose send returns one reply and whose polls follow a script."""

    def __init__(
        self, reply: Task | Message | Exception, polls: list[Task | Exception]
    ) -> None:
        self._reply = reply
        self._polls = list(polls)
        self.sent: list[SendMessageRequest] = []
        self.polled: list[str] = []

    async def send_message(self, request: SendMessageRequest):
        self.sent.append(request)
        if isinstance(self._reply, Exception):
            raise self._reply
        if isinstance(self._reply, Message):
            yield StreamResponse(message=self._reply)
        else:
            yield StreamResponse(task=self._reply)

    async def get_task(self, request: GetTaskRequest) -> Task:
        self.polled.append(request.id)
        if not self._polls:
            raise AssertionError("polled more often than scripted")
        result = self._polls.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _message() -> Message:
    return Message(role=Role.ROLE_USER, parts=[Part(text="build the report")])


def _http_error(status: int) -> A2AClientError:
    """What the JSON-RPC transport raises for an HTTP error status."""
    request = httpx.Request("POST", "https://agent.test/")
    cause = httpx.HTTPStatusError(
        "error", request=request, response=httpx.Response(status, request=request)
    )
    error = A2AClientError(f"HTTP Error: {status}")
    error.__cause__ = cause
    return error


def _network_error() -> A2AClientError:
    error = A2AClientError("Network communication error")
    error.__cause__ = httpx.ConnectError("refused")
    return error


# --- send_and_wait outside a graph -------------------------------------------


async def test_send_asks_the_agent_to_return_immediately(clock: _Clock) -> None:
    client = _ScriptedClient(_completed(), [])

    await send_and_wait(client, _message(), NO_INLINE_WAIT)  # type: ignore[arg-type]

    assert client.sent[0].configuration.return_immediately is True


async def test_message_reply_returns_without_polling(clock: _Clock) -> None:
    reply = Message(role=Role.ROLE_AGENT, parts=[Part(text="hi")], context_id="c")
    client = _ScriptedClient(reply, [])

    outcome = await send_and_wait(client, _message(), NO_INLINE_WAIT)  # type: ignore[arg-type]

    assert isinstance(outcome.reply, Message)
    assert outcome.timed_out is False
    assert client.polled == []


async def test_inline_wait_returns_a_task_that_settles_in_the_job(
    clock: _Clock,
) -> None:
    settings = A2aPollSettings(
        initial_inline_wait=timedelta(seconds=30),
        inline_poll_interval=timedelta(seconds=10),
    )
    client = _ScriptedClient(_working(), [_working(), _completed()])

    outcome = await send_and_wait(client, _message(), settings)  # type: ignore[arg-type]

    assert isinstance(outcome.reply, Task)
    assert outcome.reply.status.state == TaskState.TASK_STATE_COMPLETED
    assert client.polled == ["task-1", "task-1"]


async def test_inline_wait_stops_at_its_limit(clock: _Clock) -> None:
    settings = A2aPollSettings(
        initial_inline_wait=timedelta(seconds=20),
        inline_poll_interval=timedelta(seconds=10),
    )
    client = _ScriptedClient(_working(), [_working(), _working()])

    outcome = await send_and_wait(client, _message(), settings)  # type: ignore[arg-type]

    # Outside a graph the call cannot suspend: the task is reported as running.
    assert outcome.timed_out is True
    assert client.polled == ["task-1", "task-1"]


# A continued task (a UiPath conversational agent keeps one task per
# conversation) can briefly still show the previous turn's finished state.
SENT_STAMP = START_TIME + timedelta(seconds=3)  # server stamp on the send reply
INLINE_5S = A2aPollSettings(
    initial_inline_wait=timedelta(seconds=120),
    inline_poll_interval=timedelta(seconds=5),
)


def _stamped(task: Task, at: datetime) -> Task:
    task.status.timestamp.FromDatetime(at)
    return task


def _turn(text: str, at: datetime) -> Task:
    return _stamped(_task(TaskState.TASK_STATE_INPUT_REQUIRED, text=text), at)


def _previous_turn() -> Task:
    return _turn("turn 1 reply", START_TIME - timedelta(minutes=5))


def _next_turn() -> Message:
    message = _message()
    message.task_id = "task-1"
    return message


def _next_turn_by_context() -> Message:
    """A conversation restored from chat history: only its id is known."""
    message = _message()
    message.context_id = "ctx-1"
    return message


def _reply_text(reply: Task | Message) -> str:
    assert isinstance(reply, Task)
    return reply.artifacts[0].parts[0].text


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(_next_turn(), id="by-task"),
        pytest.param(_next_turn_by_context(), id="by-context"),
    ],
)
async def test_continued_task_skips_the_previous_turns_finished_state(
    clock: _Clock, message: Message
) -> None:
    answer = _turn("turn 2 reply", SENT_STAMP + timedelta(seconds=10))
    client = _ScriptedClient(
        _stamped(_working(), SENT_STAMP), [_previous_turn(), answer]
    )

    outcome = await send_and_wait(client, message, INLINE_5S)  # type: ignore[arg-type]

    assert _reply_text(outcome.reply) == "turn 2 reply"
    assert client.polled == ["task-1", "task-1"]


@pytest.mark.parametrize(
    ("message", "send_stamped", "poll_stamped"),
    [
        pytest.param(_message(), True, True, id="new-task"),
        pytest.param(_next_turn(), False, True, id="send-reply-unstamped"),
        pytest.param(_next_turn(), True, False, id="poll-unstamped"),
    ],
)
async def test_earlier_finished_state_is_taken_when_it_cannot_be_the_previous_turn(
    clock: _Clock, message: Message, send_stamped: bool, poll_stamped: bool
) -> None:
    reply = _stamped(_working(), SENT_STAMP) if send_stamped else _working()
    polled = (
        _previous_turn()
        if poll_stamped
        else _task(TaskState.TASK_STATE_INPUT_REQUIRED, text="turn 1 reply")
    )
    client = _ScriptedClient(reply, [polled])

    outcome = await send_and_wait(client, message, INLINE_5S)  # type: ignore[arg-type]

    assert _reply_text(outcome.reply) == "turn 1 reply"
    assert client.polled == ["task-1"]


async def test_previous_turn_check_ends_a_minute_after_the_send(
    clock: _Clock,
) -> None:
    # A finished state still stamped before the send a minute later is taken:
    # the check can delay the call, never stall it.
    client = _ScriptedClient(
        _stamped(_working(), SENT_STAMP), [_previous_turn() for _ in range(12)]
    )

    outcome = await send_and_wait(client, _next_turn(), INLINE_5S)  # type: ignore[arg-type]

    assert _reply_text(outcome.reply) == "turn 1 reply"
    assert len(client.polled) == 12


async def test_previous_turn_seen_until_the_wait_ends_is_not_reported(
    clock: _Clock,
) -> None:
    settings = A2aPollSettings(
        initial_inline_wait=timedelta(seconds=10),
        inline_poll_interval=timedelta(seconds=5),
    )
    client = _ScriptedClient(
        _stamped(_working(), SENT_STAMP), [_previous_turn(), _previous_turn()]
    )

    outcome = await send_and_wait(client, _next_turn(), settings)  # type: ignore[arg-type]

    # Reported as still running, not with the previous turn's reply.
    assert outcome.timed_out is True
    assert isinstance(outcome.reply, Task)
    assert outcome.reply.status.state == TaskState.TASK_STATE_WORKING
    assert not outcome.reply.artifacts


INLINE_30S = A2aPollSettings(
    initial_inline_wait=timedelta(seconds=30),
    inline_poll_interval=timedelta(seconds=10),
)


@pytest.mark.parametrize(
    ("error", "error_class"),
    [
        (UnsupportedOperationError("no tasks/get"), UnsupportedOperationError),
        (_http_error(401), A2AError),
        (ParseError('Task has no field named "foo".'), A2AError),
    ],
)
async def test_in_job_poll_error_a_retry_cannot_fix_fails_the_call(
    clock: _Clock, error: Exception, error_class: type[A2AError]
) -> None:
    client = _ScriptedClient(_working(), [error])

    with pytest.raises(error_class) as raised:
        await send_and_wait(client, _message(), INLINE_30S)  # type: ignore[arg-type]

    assert type(raised.value) is error_class
    assert len(client.sent) == 1


async def test_task_lost_during_the_in_job_wait_is_not_reported_as_not_found(
    clock: _Clock,
) -> None:
    client = _ScriptedClient(_working(), [TaskNotFoundError("expired")])

    with pytest.raises(A2aTaskLostError, match="task-1"):
        await send_and_wait(client, _message(), INLINE_30S)  # type: ignore[arg-type]


async def test_transient_in_job_poll_error_leaves_the_task_to_the_suspended_polls(
    clock: _Clock,
) -> None:
    client = _ScriptedClient(_working(), [_network_error()])

    outcome = await send_and_wait(client, _message(), INLINE_30S)  # type: ignore[arg-type]

    # Outside a graph the call cannot suspend: the task is reported as running.
    assert outcome.timed_out is True
    assert client.polled == ["task-1"]


async def test_send_error_keeps_its_a2a_type(clock: _Clock) -> None:
    client = _ScriptedClient(TaskNotFoundError("gone"), [])

    with pytest.raises(TaskNotFoundError, match="gone"):
        await send_and_wait(client, _message(), NO_INLINE_WAIT)  # type: ignore[arg-type]


async def test_other_send_errors_become_a2a_errors(clock: _Clock) -> None:
    client = _ScriptedClient(ConnectionError("refused"), [])

    with pytest.raises(A2AError, match="refused"):
        await send_and_wait(client, _message(), NO_INLINE_WAIT)  # type: ignore[arg-type]


def test_poll_interval_grows_then_repeats() -> None:
    settings = A2aPollSettings()
    assert [settings.poll_interval(i).total_seconds() / 60 for i in range(6)] == [
        5,
        10,
        15,
        30,
        30,
        30,
    ]


# --- send_and_wait in a graph: suspend and replay ------------------------------


class _State(TypedDict):
    result: Any


def _graph(
    client: _ScriptedClient,
    settings: A2aPollSettings,
    message: Message | None = None,
) -> Any:
    async def node(state: _State) -> dict[str, Any]:
        try:
            outcome = await send_and_wait(client, message or _message(), settings)  # type: ignore[arg-type]
        except A2AError as e:
            return {"result": {"error": type(e).__name__, "message": str(e)}}
        reply = outcome.reply
        assert isinstance(reply, Task)
        return {
            "result": {
                "state": TaskState.Name(reply.status.state),
                "timed_out": outcome.timed_out,
            }
        }

    builder = StateGraph(_State)
    builder.add_node("call", node)
    builder.add_edge(START, "call")
    builder.add_edge("call", END)
    return builder.compile(checkpointer=MemorySaver())


def _wait_until(result: dict[str, Any]) -> tuple[str, WaitUntil]:
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    value = interrupts[0].value
    assert isinstance(value, WaitUntil)
    return interrupts[0].id, value


def _timer_fired(pending: tuple[str, WaitUntil]) -> Command[Any]:
    """Resume as the runtime does when the Timer trigger fires: by interrupt id."""
    interrupt_id, wait = pending
    return Command(resume={interrupt_id: {"resumeTime": wait.resume_time.isoformat()}})


async def test_running_task_suspends_until_the_timer_then_returns(
    clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=long_running.__name__)
    client = _ScriptedClient(_working(), [_working(), _working(), _completed()])
    graph = _graph(client, NO_INLINE_WAIT)
    config: Any = {"configurable": {"thread_id": "t1"}}

    first = await graph.ainvoke({"result": None}, config)
    wait = _wait_until(first)
    assert wait[1].resume_time == START_TIME + timedelta(minutes=5)

    clock.advance(timedelta(minutes=5))
    second = await graph.ainvoke(_timer_fired(wait), config)
    wait = _wait_until(second)
    assert wait[1].resume_time == clock.now + timedelta(minutes=10)

    clock.advance(timedelta(minutes=10))
    final = await graph.ainvoke(_timer_fired(wait), config)

    assert final["result"] == {"state": "TASK_STATE_COMPLETED", "timed_out": False}
    # Sent once across three runs; each run polled exactly once.
    assert len(client.sent) == 1
    assert client.polled == ["task-1", "task-1", "task-1"]
    # Resumed runs replay the send from the checkpoint, so it is not logged again.
    sends = [r for r in caplog.records if r.getMessage().startswith("A2A send:")]
    assert len(sends) == 1


async def test_previous_turn_seen_by_a_poll_step_suspends_instead(
    clock: _Clock,
) -> None:
    answer = _stamped(_completed(), SENT_STAMP + timedelta(minutes=2))
    client = _ScriptedClient(
        _stamped(_working(), SENT_STAMP), [_previous_turn(), answer]
    )
    graph = _graph(client, NO_INLINE_WAIT, _next_turn())
    config: Any = {"configurable": {"thread_id": "t1"}}

    first = await graph.ainvoke({"result": None}, config)
    wait = _wait_until(first)

    clock.advance(timedelta(minutes=5))
    final = await graph.ainvoke(_timer_fired(wait), config)

    assert final["result"] == {"state": "TASK_STATE_COMPLETED", "timed_out": False}
    assert client.polled == ["task-1", "task-1"]


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (InvalidRequestError("bad request"), "InvalidRequestError"),
        (UnsupportedOperationError("no tasks/get"), "UnsupportedOperationError"),
        (_http_error(401), "A2AError"),
        (_http_error(403), "A2AError"),
        (_http_error(404), "A2AError"),
        # Errors the SDK does not wrap, such as a reply that does not parse.
        (ParseError('Task has no field named "foo".'), "A2AError"),
        (RuntimeError("unexpected"), "A2AError"),
    ],
)
async def test_poll_error_a_retry_cannot_fix_fails_the_call(
    clock: _Clock, error: Exception, error_type: str
) -> None:
    client = _ScriptedClient(_working(), [_working(), error])
    graph = _graph(client, NO_INLINE_WAIT)
    config: Any = {"configurable": {"thread_id": "fatal"}}

    wait = _wait_until(await graph.ainvoke({"result": None}, config))
    final = await graph.ainvoke(_timer_fired(wait), config)

    assert final["result"] == {"error": error_type, "message": str(error)}
    assert len(client.sent) == 1


async def test_task_lost_while_waiting_is_not_reported_as_not_found(
    clock: _Clock,
) -> None:
    # The agent did get the message: a caller that resends on a not-found
    # send must not take this for one.
    client = _ScriptedClient(_working(), [_working(), TaskNotFoundError("expired")])
    graph = _graph(client, NO_INLINE_WAIT)
    config: Any = {"configurable": {"thread_id": "lost"}}

    wait = _wait_until(await graph.ainvoke({"result": None}, config))
    final = await graph.ainvoke(_timer_fired(wait), config)

    assert final["result"]["error"] == "A2aTaskLostError"
    assert "task-1" in final["result"]["message"]
    assert len(client.sent) == 1


@pytest.mark.parametrize(
    "error",
    [
        InternalError("internal"),
        A2AClientTimeoutError("timed out"),
        _network_error(),
        _http_error(502),
        _http_error(429),
    ],
)
async def test_transient_poll_error_is_retried_on_the_next_wake_up(
    clock: _Clock, error: Exception
) -> None:
    client = _ScriptedClient(_working(), [error, _completed()])
    graph = _graph(client, NO_INLINE_WAIT)
    config: Any = {"configurable": {"thread_id": "transient"}}

    wait = _wait_until(await graph.ainvoke({"result": None}, config))
    final = await graph.ainvoke(_timer_fired(wait), config)

    assert final["result"] == {"state": "TASK_STATE_COMPLETED", "timed_out": False}


async def test_task_still_running_at_max_wait_is_reported(clock: _Clock) -> None:
    settings = A2aPollSettings(
        initial_inline_wait=timedelta(0),
        poll_intervals=(timedelta(minutes=40),),
        max_wait=timedelta(hours=1),
    )
    client = _ScriptedClient(_working(), [_working(), _working(), _working()])
    graph = _graph(client, settings)
    config: Any = {"configurable": {"thread_id": "t4"}}

    wait = _wait_until(await graph.ainvoke({"result": None}, config))
    clock.advance(timedelta(minutes=40))
    wait = _wait_until(await graph.ainvoke(_timer_fired(wait), config))
    # The last wait is cut to the deadline.
    assert wait[1].resume_time == START_TIME + timedelta(hours=1)

    clock.advance(timedelta(minutes=20))
    final = await graph.ainvoke(_timer_fired(wait), config)

    assert final["result"] == {"state": "TASK_STATE_WORKING", "timed_out": True}
    assert len(client.polled) == 3


# --- the A2A tool in the tool node --------------------------------------------


def _a2a_resource() -> AgentA2aResourceConfig:
    return AgentA2aResourceConfig(
        id="resource-id",
        name="Report Agent",
        description="Builds reports",
        slug="report-agent",
        folder_path="Shared",
        cached_agent_card={
            "url": "https://example.com/a2a",
            "name": "Report Agent",
            "version": "1.0.0",
            "capabilities": {},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
        },
    )


async def test_a2a_tool_resumes_and_answers_with_the_finished_task(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    monkeypatch.setattr(long_running, "DEFAULT_POLL_SETTINGS", NO_INLINE_WAIT)
    client = _ScriptedClient(_working(), [_working(), _completed("Q3 report")])
    tools, clients = create_a2a_tools_and_clients([_a2a_resource()])
    tool = tools[0]

    async def _get() -> _ScriptedClient:
        return client

    monkeypatch.setattr(clients[0], "get", _get)

    builder = StateGraph(AgentGraphState)
    builder.add_node("tool", create_tool_node(tools)[tool.name])
    builder.add_edge(START, "tool")
    builder.add_edge("tool", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config: Any = {"configurable": {"thread_id": "t5"}}
    call = {"name": tool.name, "args": {"message": "build it"}, "id": "call-1"}

    first = await graph.ainvoke(
        AgentGraphState(messages=[AIMessage(content="", tool_calls=[call])]), config
    )
    wait = _wait_until(first)
    clock.advance(timedelta(minutes=5))
    final = await graph.ainvoke(_timer_fired(wait), config)

    reply = json.loads(final["messages"][-1].content)
    assert reply == {"agent_response": "Q3 report", "task_state": "completed"}
    assert len(client.sent) == 1
    stored = final["inner_state"].tools_storage[tool.name]
    assert stored == {"task_id": None, "context_id": "ctx-1"}


def _conversational_tool_graph(
    monkeypatch: pytest.MonkeyPatch, client: _ScriptedClient
) -> tuple[Any, str]:
    """A single-conversation tool (the default) in a tool node, and its name."""
    monkeypatch.setattr(long_running, "DEFAULT_POLL_SETTINGS", NO_INLINE_WAIT)
    resource = AgentConversationalAgentToolResourceConfig(
        id="conv-id",
        name="Support Agent",
        description="Answers support questions",
        type=AgentToolType.CONVERSATIONAL_AGENT,
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        properties=AgentConversationalAgentToolProperties(
            process_name="My Conversational Agent", folder_path="Shared"
        ),
    )
    tools, clients = create_conversational_agent_tools_and_clients(
        [resource], is_conversational=False
    )
    tool = tools[0]

    async def _get() -> _ScriptedClient:
        return client

    monkeypatch.setattr(clients[0], "get", _get)

    builder = StateGraph(AgentGraphState)
    builder.add_node("tool", create_tool_node(tools)[tool.name])
    builder.add_edge(START, "tool")
    builder.add_edge("tool", END)
    return builder.compile(checkpointer=MemorySaver()), tool.name


def _call_state(tool_name: str, storage: dict[str, Any] | None = None) -> Any:
    call = {"name": tool_name, "args": {"message": "close it"}, "id": "call-1"}
    state = AgentGraphState(messages=[AIMessage(content="", tool_calls=[call])])
    if storage is not None:
        state.inner_state.tools_storage[tool_name] = storage
    return state


async def test_conversational_tool_suspends_instead_of_reporting_a_failure(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    """The suspend passes through the tool's request-error handling."""
    client = _ScriptedClient(_working(), [_working(), _completed("ticket closed")])
    graph, tool_name = _conversational_tool_graph(monkeypatch, client)
    config: Any = {"configurable": {"thread_id": "t6"}}

    first = await graph.ainvoke(_call_state(tool_name), config)
    final = await graph.ainvoke(_timer_fired(_wait_until(first)), config)

    reply = json.loads(final["messages"][-1].content)
    assert reply["agent_response"] == "ticket closed"
    assert reply["task_state"] == "completed"
    assert len(client.sent) == 1


async def test_conversational_tool_does_not_resend_when_the_task_is_lost_while_waiting(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    # A not-found on the send restarts the conversation and resends the message.
    # On a poll the agent already has the message, so it must not be sent again.
    client = _ScriptedClient(_working(), [_working(), TaskNotFoundError("expired")])
    graph, tool_name = _conversational_tool_graph(monkeypatch, client)
    config: Any = {"configurable": {"thread_id": "t7"}}
    storage = {
        "current": "c1",
        "conversations": {"c1": {"task_id": "c1", "state": "open", "last_used": 1}},
    }

    first = await graph.ainvoke(_call_state(tool_name, storage), config)
    final = await graph.ainvoke(_timer_fired(_wait_until(first)), config)

    reply = json.loads(final["messages"][-1].content)
    assert reply["error"] == "request_failed"
    assert "received the message" in reply["message"]
    assert len(client.sent) == 1
