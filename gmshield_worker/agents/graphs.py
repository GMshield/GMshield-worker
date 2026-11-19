"""Agent graphs (Story 2.10): every agent is a LangGraph graph compiled by `compile_graph`, and only there.

A graph lives in memory for one job: it is compiled without persistence, shared state or a pause point, so nothing of
a job outlives its attempt and a retry starts afresh (a boundary test fails the build if anything in the worker adds
one). `run_graph` runs it with a small step limit and no callback handler of ours.

`AnswerWithTools` is the node body most agents share: it calls the model through `models.output.stream_call`, runs the
read-only tools the model asks for, gives their results back, and stops at the first answer without a tool call, at the
first failure, or after `max_calls` model calls (then `model.truncated`: the answer was cut by a limit). `should_stop`
(Story 2.6: the GM's stop) is asked before each model call and, inside `stream_call`, before each chunk: once true, the
node ends `stopped`, with no failure. Library text enters only as tool results or user-turn data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, message_chunk_to_message
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from gmshield_worker.models.output import TRUNCATED, CitationMarkers, Emit, ModelFailed, stream_call

# Steps of one run: a one-node graph needs two; more means a graph loops.
RECURSION_LIMIT = 10
DEFAULT_MAX_CALLS = 4


class AnswerState(TypedDict, total=False):
    """The state of an answering graph: the conversation so far, how it ended (a failure's code, or stopped), and how
    many model calls it took."""

    messages: list[BaseMessage]
    failed: str | None
    stopped: bool
    calls: int


def compile_graph(builder: StateGraph[Any, Any, Any, Any]) -> CompiledStateGraph[Any, Any, Any, Any]:
    """The compiled graph: in memory for one job, with LangGraph's defaults for everything that would keep or pause it
    (nothing is kept, nothing pauses)."""
    return builder.compile()


def run_graph(graph: CompiledStateGraph[Any, Any, Any, Any], state: AnswerState) -> AnswerState:
    """Runs the graph once to its end, within `RECURSION_LIMIT` steps, adding no callback handler of ours. LangChain may
    still add its own tracer from the environment: trace export is refused at start by `settings.check_tracing`."""
    result = graph.invoke(state, config={"recursion_limit": RECURSION_LIMIT, "callbacks": []})
    return cast(AnswerState, result)


@dataclass
class AnswerWithTools:
    """The node body: model, then the tools it asked for, until it answers. Events go to `emit` as they come."""

    model: BaseChatModel
    tools: Sequence[BaseTool]
    markers: CitationMarkers
    emit: Emit
    max_calls: int = DEFAULT_MAX_CALLS
    should_stop: Callable[[], bool] = field(default=lambda: False)

    def __call__(self, state: AnswerState) -> AnswerState:
        messages = list(state.get("messages") or [])
        bound = self.model.bind_tools(list(self.tools)) if self.tools else self.model
        by_name = {tool.name: tool for tool in self.tools}
        calls = 0
        while calls < self.max_calls and not self.should_stop():
            calls += 1
            result = stream_call(bound, messages, markers=self.markers, emit=self.emit, should_stop=self.should_stop)
            if result.message is None:
                return {"messages": messages, "failed": result.failed, "stopped": result.stopped, "calls": calls}
            answer = message_chunk_to_message(result.message)
            messages.append(answer)
            if not isinstance(answer, AIMessage) or not answer.tool_calls:
                return {"messages": messages, "failed": None, "stopped": False, "calls": calls}
            for call in answer.tool_calls:
                tool = by_name.get(call["name"])
                if tool is None:
                    messages.append(ToolMessage(content="unknown tool", tool_call_id=call["id"] or "", status="error"))
                    continue
                messages.append(tool.invoke(call))
        if self.should_stop():
            return {"messages": messages, "failed": None, "stopped": True, "calls": calls}
        self.emit(ModelFailed(TRUNCATED))
        return {"messages": messages, "failed": TRUNCATED, "stopped": False, "calls": calls}


def answer_graph(node: Callable[[AnswerState], AnswerState]) -> CompiledStateGraph[Any, Any, Any, Any]:
    """A one-node graph around an answering node."""
    builder: StateGraph[Any, Any, Any, Any] = StateGraph(AnswerState)
    builder.add_node("answer", cast(Any, node))
    builder.set_entry_point("answer")
    builder.set_finish_point("answer")
    return compile_graph(builder)
