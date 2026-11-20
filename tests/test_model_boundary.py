"""The model boundary (Story 2.10), checked on the source tree and the installed packages:

- provider SDKs and LangChain's provider packages are imported only inside `gmshield_worker/models/`;
- nothing in `gmshield_worker/` keeps or pauses a graph: no checkpointer, `MemorySaver`, LangGraph store or
  interrupt, and every graph is compiled by `agents.graphs.compile_graph`;
- the image holds no OpenTelemetry SDK.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import re
from pathlib import Path

import pytest

from gmshield_worker.agents import graphs

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "gmshield_worker"
MODELS = PACKAGE / "models"
GENERATED = PACKAGE / "backend" / "generated"
PROVIDER_PACKAGES = frozenset({"openai", "anthropic", "langchain_openai", "langchain_anthropic"})
KEEPING_MODULES = ("langgraph.checkpoint", "langgraph.store", "langgraph.cache")
KEEPING_NAMES = frozenset(
    {"MemorySaver", "InMemorySaver", "BaseCheckpointSaver", "InMemoryStore", "BaseStore", "interrupt", "Command"}
)
KEEPING_WORDS = re.compile(r"\b(checkpointer|MemorySaver|InMemorySaver|interrupt|InMemoryStore|BaseStore)\b")


def sources() -> list[Path]:
    return [path for path in sorted(PACKAGE.rglob("*.py")) if GENERATED not in path.parents]


def graph_compile(node: ast.AST) -> bool:
    """A `.compile(...)` call on anything but the `re` module."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compile"
        and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "re")
    )


def imported_modules(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            found.append(node.module)
    return found


def test_provider_packages_are_imported_only_inside_models() -> None:
    offenders = []
    for path in sources():
        if MODELS in path.parents:
            continue
        for module in imported_modules(ast.parse(path.read_text(encoding="utf-8"))):
            if module.split(".")[0] in PROVIDER_PACKAGES:
                offenders.append(f"{path.relative_to(ROOT)}: {module}")
    assert offenders == []
    # The check sees them where they are.
    inside = {
        module.split(".")[0]
        for path in MODELS.glob("*.py")
        for module in imported_modules(ast.parse(path.read_text(encoding="utf-8")))
    }
    assert inside >= PROVIDER_PACKAGES


def test_nothing_in_the_worker_keeps_or_pauses_a_graph() -> None:
    offenders = []
    for path in sources():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        name = str(path.relative_to(ROOT))
        offenders += [f"{name}: word {match[0]}" for match in KEEPING_WORDS.finditer(text)]
        offenders += [f"{name}: {module}" for module in imported_modules(tree) if module.startswith(KEEPING_MODULES)]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("langgraph"):
                offenders += [f"{name}: {alias.name}" for alias in node.names if alias.name in KEEPING_NAMES]
            if graph_compile(node):
                assert isinstance(node, ast.Call)
                # LangGraph's defaults keep nothing and never pause: a graph is compiled without any argument.
                offenders += [f"{name}: compile with arguments" for _ in [*node.args, *node.keywords]]
    assert offenders == []


def test_graphs_are_compiled_only_by_compile_graph() -> None:
    compiling = []
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if graph_compile(node):
                compiling.append(str(path.relative_to(ROOT)))
    assert compiling == ["gmshield_worker/agents/graphs.py"]


def test_a_compiled_graph_keeps_nothing_and_never_pauses() -> None:
    graph = graphs.answer_graph(lambda state: state)
    assert graph.checkpointer is None
    assert graph.store is None
    assert not graph.interrupt_before_nodes
    assert not graph.interrupt_after_nodes
    assert graphs.run_graph(graph, {"messages": [], "calls": 0}) == {"messages": [], "calls": 0}


@pytest.mark.parametrize("module", ["opentelemetry.sdk", "opentelemetry.exporter"])
def test_no_opentelemetry_sdk_or_exporter_is_installed(module: str) -> None:
    package = module.split(".", maxsplit=1)[0]
    assert importlib.util.find_spec(package) is None or importlib.util.find_spec(module) is None


def test_no_opentelemetry_sdk_distribution_is_installed_or_locked() -> None:
    for distribution in ("opentelemetry-sdk", "opentelemetry-exporter-otlp", "opentelemetry-exporter-otlp-proto-http"):
        with pytest.raises(importlib.metadata.PackageNotFoundError):
            importlib.metadata.distribution(distribution)
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "opentelemetry-sdk"' not in lock
    assert 'name = "opentelemetry-exporter-' not in lock
