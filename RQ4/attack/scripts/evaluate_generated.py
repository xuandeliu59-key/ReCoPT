#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ATTACK_CODE_ROOT = Path(
    os.environ.get("RQ4_ATTACK_CODE_ROOT", str(ROOT / "attack_code"))
).expanduser().resolve()
sys.path.insert(0, str(ATTACK_CODE_ROOT))

from rq4_attack import config as path_config
from rq4_attack.config import data_path, model_paths, output_dir
from rq4_attack.data import atomic_json, label_of, read_jsonl
from rq4_attack.models import RetrievalVictim, device_from_name, load_retrieval


def rows(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def clone_correct(
    victim: RetrievalVictim,
    query_rows: list[dict],
    indexes: dict[str, int],
    batch_size: int,
) -> np.ndarray:
    if not query_rows:
        return np.zeros(0, dtype=bool)
    vectors = victim.embed([row["code"] for row in query_rows], batch_size=batch_size)
    scores = vectors @ victim.gallery_vectors.T
    labels = np.asarray([label_of(row) for row in query_rows], dtype=np.int64)
    for row_index, row in enumerate(query_rows):
        row_uid = str(row["submission_id"])
        if row_uid in indexes:
            scores[row_index, indexes[row_uid]] = -1.0e30
    predicted = victim.gallery_labels[np.argmax(scores, axis=1)]
    return predicted == labels


def release(victim, model, tokenizer) -> None:
    del victim, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_clone(
    shift: str,
    attack: str,
    model_name: str,
    setting: str,
    device: torch.device,
    batch_size: int,
) -> dict:
    attacked_model = model_name if setting == "adaptive" else "Original"
    branch = output_dir(
        "adaptive", attack, "clone_detection", shift, attacked_model
    )
    clean = rows(branch / "clean.jsonl")
    adversarial = rows(branch / "adversarial.jsonl")
    if [str(row["submission_id"]) for row in clean] != [
        str(row["submission_id"]) for row in adversarial
    ]:
        raise RuntimeError(f"clean/adversarial UID mismatch: {branch}")

    source = rows(data_path("clone_detection", shift, "id_test"))
    ids = [str(row["submission_id"]) for row in source]
    indexes = {row_uid: index for index, row_uid in enumerate(ids)}
    model, tokenizer = load_retrieval(
        model_paths("clone_detection", shift, model_name), device
    )
    victim = RetrievalVictim(
        model,
        tokenizer,
        device,
        [row["code"] for row in source],
        [label_of(row) for row in source],
        ids,
        batch_size=batch_size,
    )
    full_clean = clone_correct(victim, source, indexes, batch_size)
    saved_clean = clone_correct(victim, clean, indexes, batch_size)
    saved_adversarial = clone_correct(victim, adversarial, indexes, batch_size)
    successes = int(np.sum(saved_clean & ~saved_adversarial))
    denominator = int(full_clean.sum()) if setting == "adaptive" else int(saved_clean.sum())
    result = {
        "task": "clone_detection",
        "shift": shift,
        "attack": attack,
        "setting": setting,
        "attacked_model": attacked_model,
        "evaluated_model": model_name,
        "base_size": len(source),
        "full_clean_correct": int(full_clean.sum()),
        "saved_attack_rows": len(adversarial),
        "clean_correct_denominator": denominator,
        "revalidated_successes": successes,
        "asr": successes / denominator if denominator else None,
    }
    release(victim, model, tokenizer)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="clone_detection")
    parser.add_argument("--shifts", default="cst,token")
    parser.add_argument("--attacks", default="itgen,coda")
    parser.add_argument("--settings", default="adaptive,transfer")
    parser.add_argument("--models", default="Original,ContraBERT_C,ReCoPT")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--train-data-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--generated-root", type=Path, default=ROOT / "generated")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "asr_results.json"
    )
    args = parser.parse_args()
    path_config.configure_paths(
        data_root=args.data_root,
        train_data_root=args.train_data_root,
        model_root=args.model_root,
        resource_root=args.resource_root,
        output_root=args.generated_root,
        results_root=args.results_root,
    )
    tasks = [value for value in args.tasks.split(",") if value]
    shifts = [value for value in args.shifts.split(",") if value]
    attacks = [value for value in args.attacks.split(",") if value]
    settings = [value for value in args.settings.split(",") if value]
    models = [value for value in args.models.split(",") if value]
    device = device_from_name(args.device)
    results = []

    for task in tasks:
        for shift in shifts:
            for attack in attacks:
                for setting in settings:
                    for model_name in models:
                        if task != "clone_detection":
                            raise ValueError(f"unsupported task: {task}")
                        result = evaluate_clone(
                            shift, attack, model_name, setting, device,
                            args.batch_size,
                        )
                        results.append(result)
                        print(
                            f"{task:18s} {shift:5s} {attack:6s} {setting:8s} "
                            f"{model_name:12s} success={result['revalidated_successes']}/"
                            f"{result['clean_correct_denominator']} ASR={result['asr']}",
                            flush=True,
                        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, results)
    print(f"RESULT_SAVED={args.output}")


if __name__ == "__main__":
    main()
