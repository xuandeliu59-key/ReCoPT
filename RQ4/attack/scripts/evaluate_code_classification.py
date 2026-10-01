#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from rq4_attack import config
from rq4_attack.data import label_of, read_jsonl
from rq4_attack.models import ClassificationVictim, device_from_name, load_classification


ATTACKS = ("itgen", "coda")
MODELS = ("Original", "ContraBERT_C", "ReCoPT")
SHIFTS = ("cst", "token")


def load(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def uid(row: dict) -> str:
    return str(row["submission_id"])


def paired(directory: Path) -> tuple[list[dict], list[dict]]:
    clean = load(directory / "clean.jsonl")
    adversarial = load(directory / "adversarial.jsonl")
    if [uid(row) for row in clean] != [uid(row) for row in adversarial]:
        raise RuntimeError(f"clean/adversarial UID mismatch: {directory}")
    return clean, adversarial


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def correctness(victim: ClassificationVictim, rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=bool)
    _, predictions = victim.predict([row["code"] for row in rows], batch_size=32)
    labels = np.asarray([label_of(row) for row in rows])
    return predictions == labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    data_root = root / "data"
    generated_root = root / "attack" / "generated"
    config.configure_paths(model_root=args.model_root.resolve())
    device = device_from_name(args.device)
    results = []

    for shift in SHIFTS:
        base = load(data_root / "stratified" / shift / "code_classification/id_test.jsonl")
        if len(base) != 2000 or len({uid(row) for row in base}) != 2000:
            raise RuntimeError(f"{shift}: invalid 2,000-row base")
        for model_name in MODELS:
            paths = config.model_paths("code_classification", shift, model_name)
            model, tokenizer = load_classification(paths, device, num_labels=250)
            victim = ClassificationVictim(model, tokenizer, device)
            base_correct = correctness(victim, base)

            for attack in ATTACKS:
                adaptive_dir = (
                    generated_root / "adaptive" / attack / "code_classification"
                    / shift / model_name
                )
                clean, adversarial = paired(adaptive_dir)
                clean_correct = correctness(victim, clean)
                adversarial_correct = correctness(victim, adversarial)
                successes = int(np.sum(clean_correct & ~adversarial_correct))
                denominator = int(base_correct.sum())
                results.append(
                    {
                        "task": "code_classification",
                        "shift": shift,
                        "attack": attack,
                        "setting": "adaptive",
                        "attacked_model": model_name,
                        "evaluated_model": model_name,
                        "base_size": len(base),
                        "saved_attack_rows": len(adversarial),
                        "clean_correct_denominator": denominator,
                        "revalidated_successes": successes,
                        "asr": successes / denominator if denominator else None,
                    }
                )

                transfer_dir = (
                    generated_root / "adaptive" / attack / "code_classification"
                    / shift / "Original"
                )
                transfer_clean, transfer_adversarial = paired(transfer_dir)
                transfer_clean_correct = correctness(victim, transfer_clean)
                transfer_adversarial_correct = correctness(victim, transfer_adversarial)
                transfer_denominator = int(transfer_clean_correct.sum())
                transfer_successes = int(
                    np.sum(transfer_clean_correct & ~transfer_adversarial_correct)
                )
                results.append(
                    {
                        "task": "code_classification",
                        "shift": shift,
                        "attack": attack,
                        "setting": "transfer",
                        "attacked_model": "Original",
                        "evaluated_model": model_name,
                        "base_size": len(base),
                        "saved_attack_rows": len(transfer_adversarial),
                        "clean_correct_denominator": transfer_denominator,
                        "revalidated_successes": transfer_successes,
                        "asr": transfer_successes / transfer_denominator if transfer_denominator else None,
                    }
                )
                print(
                    f"{shift} {attack} {model_name}: "
                    f"adaptive={successes}/{denominator} "
                    f"transfer={transfer_successes}/{transfer_denominator}",
                    flush=True,
                )
            del victim, model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output = root / "results" / "code_classification_asr_results.json"
    write_json(output, results)
    print(f"RESULT_SAVED={output}")


if __name__ == "__main__":
    main()
