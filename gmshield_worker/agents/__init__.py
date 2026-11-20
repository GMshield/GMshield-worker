"""Agents (Story 2.10 onward): the model-driven steps that answer a GM's questions from retrieved passages.

Every agent is a LangGraph graph compiled through `graphs.compile_graph`, in memory for one job and kept nowhere.
Its tools (`tools.py`) are read-only calls of the internal API under the job's grant. Library text reaches a model only
as a tool result or user-turn data, with its passage id and section titles, never in the system prompt. Models come
from `gmshield_worker.models` only.
"""
