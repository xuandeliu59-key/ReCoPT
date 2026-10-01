from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: Path, limit: int | None = None) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        count = 0
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "code" not in row or "label" not in row:
                raise ValueError(f"{path}:{line_number}: expected code and label")
            yield row
            count += 1
            if limit is not None and count >= limit:
                return


def label_of(row: dict) -> int:
    value = row.get("rq4_label", row["label"])
    if isinstance(value, str) and value.startswith("label_"):
        value = value[6:]
    return int(value)


def sample_id(row: dict, index: int) -> str:
    value = str(row.get("submission_id", row.get("idx", index)))
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def normalized_sha256(code: str) -> str:
    normalized = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

