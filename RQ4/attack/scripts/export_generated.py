#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "runtime" / "raw" / "adaptive"
GENERATED_ROOT = ROOT / "generated" / "adaptive"
ATTACKS = ("itgen", "coda")
TASKS = ("clone_detection",)
SHIFTS = ("cst", "token")
MODELS = ("Original", "ContraBERT_C", "ReCoPT")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_directory(destination: Path, build) -> None:
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    build(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def export_clone(attack: str, shift: str, model: str) -> None:
    source = RAW_ROOT / attack / "clone_detection" / shift / model
    clean = read_jsonl(source / "clean.jsonl")
    adversarial = read_jsonl(source / "adversarial.jsonl")
    clean_ids = [str(row["submission_id"]) for row in clean]
    adversarial_ids = [str(row["submission_id"]) for row in adversarial]
    if clean_ids != adversarial_ids:
        raise RuntimeError(f"clean/adversarial UID mismatch: {source}")

    destination = GENERATED_ROOT / attack / "clone_detection" / shift / model

    def build(directory: Path) -> None:
        write_jsonl(directory / "clean.jsonl", clean)
        write_jsonl(directory / "adversarial.jsonl", adversarial)

    atomic_directory(destination, build)
    print(f"{attack}/clone_detection/{shift}/{model}: exported={len(adversarial)}")


def main() -> None:
    global RAW_ROOT, GENERATED_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--generated-root", type=Path, default=GENERATED_ROOT)
    parser.add_argument("--attacks", default=",".join(ATTACKS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--shifts", default=",".join(SHIFTS))
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()

    RAW_ROOT = args.raw_root.expanduser().resolve()
    GENERATED_ROOT = args.generated_root.expanduser().resolve()
    attacks = [value for value in args.attacks.split(",") if value]
    tasks = [value for value in args.tasks.split(",") if value]
    shifts = [value for value in args.shifts.split(",") if value]
    models = [value for value in args.models.split(",") if value]
    for attack in attacks:
        for shift in shifts:
            for model in models:
                if "clone_detection" in tasks:
                    export_clone(attack, shift, model)


if __name__ == "__main__":
    main()
