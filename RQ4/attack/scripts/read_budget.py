#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("clone_detection",), required=True)
    parser.add_argument("--attack", choices=("itgen", "coda"), required=True)
    parser.add_argument(
        "--model", choices=("Original", "ContraBERT_C", "ReCoPT"), required=True
    )
    parser.add_argument("--settings", type=Path, default=ROOT / "settings.json")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    budget = settings["budgets"][args.task][args.model]
    candidates = budget[f"{args.attack}_candidates"]
    common = settings["common"]
    print(
        budget["query_budget"],
        candidates,
        budget["coda_references"],
        common["time_limit_seconds"],
        common["batch_size"],
    )


if __name__ == "__main__":
    main()
