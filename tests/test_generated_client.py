"""The generated client matches contracts/internal.json, and no hand-written type describes an
internal request or response."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from gmshield_worker.backend.generated import _contract
from gmshield_worker.backend.generated.api.contract import get_internal_contract_version

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "contracts" / "internal.json"
GENERATED = ROOT / "gmshield_worker" / "backend" / "generated"


def snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def test_the_client_was_generated_from_this_contract() -> None:
    document = CONTRACT.read_bytes()
    assert json.loads(document)["info"]["version"] == _contract.CONTRACT_VERSION
    assert hashlib.sha256(document).hexdigest() == _contract.CONTRACT_SHA256


def test_every_operation_has_a_generated_module() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for path_item in contract["paths"].values():
        for operation in path_item.values():
            tag = snake(operation["tags"][0])
            module = GENERATED / "api" / tag / f"{snake(operation['operationId'])}.py"
            assert module.is_file(), module
    assert callable(get_internal_contract_version.sync_detailed)


def test_every_schema_has_a_generated_model() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for schema in contract["components"]["schemas"]:
        assert (GENERATED / "models" / f"{snake(schema)}.py").is_file(), schema


def test_no_hand_written_class_describes_a_backend_schema() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    schemas = set(contract["components"]["schemas"])
    for source in (ROOT / "gmshield_worker").rglob("*.py"):
        if GENERATED in source.parents:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        assert not classes & schemas, source
