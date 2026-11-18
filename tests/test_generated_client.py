"""The generated client matches contracts/internal.json, and no hand-written type describes an
internal request or response."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from openapi_python_client.strings import PythonIdentifier, snake_case

from gmshield_worker.backend.generated import _contract
from gmshield_worker.backend.generated.api.contract import get_internal_contract_version

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "contracts" / "internal.json"
GENERATED = ROOT / "gmshield_worker" / "backend" / "generated"
HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")


def operations(contract: dict[str, object]) -> list[dict[str, object]]:
    """Every operation of the contract; path-level keys such as `parameters` are not operations."""
    paths = contract["paths"]
    assert isinstance(paths, dict)
    return [item[method] for item in paths.values() for method in HTTP_METHODS if method in item]


def tag_module(operation: dict[str, object]) -> str:
    """The folder the generator puts an operation in: its first tag, or `default` (as the generator)."""
    tags = operation.get("tags") or ["default"]
    assert isinstance(tags, list)
    return str(PythonIdentifier(value=str(tags[0]), prefix="tag"))


def test_the_client_was_generated_from_this_contract() -> None:
    document = CONTRACT.read_bytes()
    assert json.loads(document)["info"]["version"] == _contract.CONTRACT_VERSION
    assert hashlib.sha256(document).hexdigest() == _contract.CONTRACT_SHA256


def test_every_operation_has_a_generated_module() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    found = operations(contract)
    assert found
    for operation in found:
        module = GENERATED / "api" / tag_module(operation) / f"{snake_case(str(operation['operationId']))}.py"
        assert module.is_file(), module
    assert callable(get_internal_contract_version.sync_detailed)


def test_every_schema_has_a_generated_model() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for schema in contract["components"]["schemas"]:
        assert (GENERATED / "models" / f"{snake_case(schema)}.py").is_file(), schema


def test_no_hand_written_class_describes_a_backend_schema() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    schemas = set(contract["components"]["schemas"])
    for source in (ROOT / "gmshield_worker").rglob("*.py"):
        if GENERATED in source.parents:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        assert not classes & schemas, source
