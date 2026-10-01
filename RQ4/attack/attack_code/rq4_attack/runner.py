from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import config as path_config
from .coda_adapter import CODAAttack, CODAResources
from .config import (
    ATTACKS, MODELS, SHIFTS, TASKS, data_path, model_paths, output_dir,
    resource_dir, validate_paths,
)
from .data import atomic_json, atomic_text, label_of, read_jsonl, sample_id
from .itgen_adapter import ITGenAttack, build_identifier_vocab, configure_itgen_root
from .java import JavaTools
from .models import (
    BinaryClassificationVictim, ClassificationVictim, RetrievalVictim, device_from_name,
    load_classification, load_retrieval,
)
from .resources import ReferenceEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_victim(task, shift, model_name, test_rows, device, batch_size):
    paths = model_paths(task, shift, model_name)
    if task in {"code_classification", "defect_detection"}:
        labels = 250 if task == "code_classification" else 1
        model, tokenizer = load_classification(paths, device, num_labels=labels)
        victim_class = ClassificationVictim if task == "code_classification" else BinaryClassificationVictim
        return victim_class(model, tokenizer, device), tokenizer
    model, tokenizer = load_retrieval(paths, device)
    victim = RetrievalVictim(
        model,
        tokenizer,
        device,
        [row["code"] for row in test_rows],
        [label_of(row) for row in test_rows],
        [sample_id(row, index) for index, row in enumerate(test_rows)],
        batch_size=batch_size,
    )
    return victim, tokenizer


def sample_directory(root: Path, index: int, row: dict) -> Path:
    return root / "samples" / f"{index:06d}_{sample_id(row, index)}"


def build_attack(args, victim, tokenizer, java, train_rows, device):
    if args.attack == "itgen":
        vocabulary = build_identifier_vocab(train_rows, java, args.vocab_limit)
        return ITGenAttack(
            victim, tokenizer, java, vocabulary,
            query_budget=args.query_budget,
            time_limit=args.time_limit,
            candidates_per_variable=args.candidates,
            eval_batch_size=args.batch_size,
            seed=args.seed,
        )
    resources = CODAResources(resource_dir(args.task, args.shift))
    reference_model = ReferenceEncoder(device)
    return CODAAttack(
        victim, java, resources, reference_model, reference_model.tokenizer,
        query_budget=args.query_budget,
        eval_batch_size=args.batch_size,
        reference_count=args.coda_references,
        candidates_per_identifier=args.candidates,
    )


def rebuild(root: Path, rows: list[dict]) -> dict:
    manifest = []
    clean_rows = []
    adversarial_rows = []
    for index, row in enumerate(rows):
        directory = sample_directory(root, index, row)
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest.append(summary)
        if summary.get("status") != "success":
            continue
        clean = dict(row)
        clean_rows.append(clean)
        attacked = dict(row)
        attacked["code"] = (directory / "adversarial.java").read_text(encoding="utf-8")
        adversarial_rows.append(attacked)
    atomic_text(
        root / "manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
    )
    atomic_text(
        root / "clean.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean_rows),
    )
    atomic_text(
        root / "adversarial.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in adversarial_rows),
    )
    counts: dict[str, int] = {}
    for row in manifest:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    aggregate = {
        "processed": len(manifest),
        "successful_sources": len(adversarial_rows),
        "status_counts": counts,
    }
    atomic_json(root / "aggregate_summary.json", aggregate)
    return aggregate


def run(args) -> None:
    path_config.configure_paths(
        data_root=args.data_root,
        train_data_root=args.train_data_root,
        model_root=args.model_root,
        resource_root=args.resource_root,
        output_root=args.output_root,
        results_root=args.results_root,
    )
    configure_itgen_root(args.itgen_root)
    if args.setting == "transfer" and args.attacked_model != "Original":
        raise ValueError("transfer attacks must target Original; other models are evaluated later")
    validate_paths(args.task, args.shift, args.attacked_model)
    set_seed(args.seed)
    device = device_from_name(args.device)
    source_path = data_path(args.task, args.shift, args.test_role)
    train_path = data_path(args.task, args.shift, "train")
    test_rows = list(read_jsonl(source_path))
    train_rows = list(read_jsonl(train_path, args.vocab_limit))
    root = output_dir(
        args.setting, args.attack, args.task, args.shift, args.attacked_model
    )
    root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "setting": args.setting,
        "attack": args.attack,
        "task": args.task,
        "shift": args.shift,
        "attacked_model": args.attacked_model,
        "source": str(source_path),
        "seed": args.seed,
        "query_budget": args.query_budget,
        "time_limit": args.time_limit,
        "candidates": args.candidates,
    }
    config_path = root / "configuration.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise RuntimeError(f"resume configuration differs: {config_path}")
    else:
        atomic_json(config_path, configuration)

    victim, tokenizer = load_victim(
        args.task, args.shift, args.attacked_model, test_rows, device, args.batch_size
    )
    java = JavaTools(args.java_language_so)
    attack = build_attack(args, victim, tokenizer, java, train_rows, device)
    stop = len(test_rows) if args.limit is None else min(len(test_rows), args.start + args.limit)
    for index in range(args.start, stop):
        row = test_rows[index]
        directory = sample_directory(root, index, row)
        complete = directory / ".complete"
        summary_path = directory / "summary.json"
        if complete.is_file() and summary_path.is_file():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        atomic_text(directory / "original.java", row["code"])
        if isinstance(victim, RetrievalVictim):
            victim.set_source(index, label_of(row))
        started = time.time()
        result = None
        error = None
        try:
            if not java.syntax_ok(row["code"]):
                status = "source_syntax_error"
            else:
                result = attack.attack(row["code"], label_of(row))
                status = (
                    "success" if result.success
                    else result.metadata.get("reason", "attack_failed")
                )
                if result.success:
                    atomic_text(directory / "adversarial.java", result.code)
        except Exception as exc:  
            status = "error"
            error = {"type": type(exc).__name__, "message": str(exc)}

        summary = {
            **configuration,
            "source_index": index,
            "submission_id": sample_id(row, index),
            "true_label": label_of(row),
            "status": status,
            "elapsed_seconds": round(time.time() - started, 3),
            "score_drop": None if result is None else result.score_drop,
            "queries": None if result is None else result.queries,
            "metadata": None if result is None else result.metadata,
            "error": error,
        }
        atomic_json(summary_path, summary)
        atomic_text(complete, status + "\n")
        print(
            f"[{index + 1}/{stop}] {summary['submission_id']} status={status}",
            flush=True,
        )
    aggregate = rebuild(root, test_rows)
    print(json.dumps(aggregate, ensure_ascii=False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run ITGen/CODA against Paper RQ1 Java models")
    parser.add_argument("--setting", choices=("transfer", "adaptive"), required=True)
    parser.add_argument("--attack", choices=ATTACKS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--shift", choices=SHIFTS, required=True)
    parser.add_argument("--attacked-model", choices=MODELS, required=True)
    parser.add_argument("--test-role", choices=("id_test", "ood_test"), default="id_test")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--query-budget", type=int, default=500)
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coda-references", type=int, default=50)
    parser.add_argument("--vocab-limit", type=int, default=10000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--train-data-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--itgen-root", type=Path)
    parser.add_argument("--java-language-so", type=Path)
    args = parser.parse_args(argv)
    for name in ("query_budget", "time_limit", "candidates", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()

