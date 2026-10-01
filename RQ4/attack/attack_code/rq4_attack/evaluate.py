from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config as path_config
from .config import ATTACKS, MODELS, SHIFTS, TASKS, model_paths, output_dir
from .data import atomic_json, label_of, read_jsonl
from .models import (
    BinaryClassificationVictim, ClassificationVictim, RetrievalVictim, device_from_name,
    load_classification, load_retrieval,
)


def classification_metrics(victim, rows, batch_size):
    if not rows:
        return {"count": 0, "accuracy": None}
    probabilities, predictions = victim.predict(
        [row["code"] for row in rows], batch_size=batch_size
    )
    labels = np.asarray([label_of(row) for row in rows])
    return {
        "count": len(rows),
        "accuracy": float(np.mean(predictions == labels)),
        "mean_true_label_probability": float(
            np.mean(probabilities[np.arange(len(rows)), labels])
        ),
    }


def retrieval_metrics(victim: RetrievalVictim, rows, batch_size):
    if not rows:
        return {
            "count": 0, "map_at_r": None, "mrr": None,
            "p_at_1": None, "p_at_5": None,
        }
    vectors = victim.embed([row["code"] for row in rows], batch_size=batch_size)
    scores = np.matmul(vectors, victim.gallery_vectors.T)
    labels = np.asarray([label_of(row) for row in rows])
    gallery_by_id = {
        str(row_id): index for index, row_id in enumerate(victim.gallery_ids)
    }
    if len(gallery_by_id) != len(victim.gallery_ids):
        raise RuntimeError("retrieval gallery contains duplicate submission_id values")
    gallery_indexes = []
    for row in rows:
        if "submission_id" not in row:
            raise KeyError("retrieval row is missing submission_id")
        row_id = str(row["submission_id"])
        if row_id not in gallery_by_id:
            raise KeyError(f"submission_id not found in retrieval gallery: {row_id}")
        gallery_indexes.append(gallery_by_id[row_id])
    aps, reciprocal, p1, p5 = [], [], [], []
    for row_index in range(len(rows)):
        gallery_index = gallery_indexes[row_index]
        if gallery_index is not None and int(gallery_index) >= 0:
            gallery_index = int(gallery_index)
            scores[row_index, gallery_index] = -1e30
        relevant = victim.gallery_labels == labels[row_index]
        relevant = relevant.copy()
        if gallery_index is not None and int(gallery_index) >= 0:
            relevant[int(gallery_index)] = False
        count = int(relevant.sum())
        if count <= 0:
            continue
        order = np.argsort(scores[row_index])[::-1]
        ranked = relevant[order]
        positions = np.flatnonzero(ranked[:count])
        precisions = [
            (rank + 1) / (position + 1) for rank, position in enumerate(positions)
        ]
        aps.append(float(sum(precisions) / count))
        first = int(np.flatnonzero(ranked)[0])
        reciprocal.append(1.0 / (first + 1))
        p1.append(float(ranked[0]))
        p5.append(float(ranked[:5].mean()))
    return {
        "count": len(rows),
        "valid_queries": len(aps),
        "map_at_r": float(np.mean(aps)) if aps else None,
        "mrr": float(np.mean(reciprocal)) if reciprocal else None,
        "p_at_1": float(np.mean(p1)) if p1 else None,
        "p_at_5": float(np.mean(p5)) if p5 else None,
    }


def adaptive_denominator(root: Path) -> dict:
    manifest_path = root / "manifest.jsonl"
    manifest = []
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = [json.loads(line) for line in handle if line.strip()]
    clean_incorrect = sum(row.get("status") == "clean_incorrect" for row in manifest)
    source_invalid = sum(row.get("status") == "source_syntax_error" for row in manifest)
    eligible = len(manifest) - clean_incorrect - source_invalid
    successes = sum(row.get("status") == "success" for row in manifest)
    return {
        "all_processed_denominator": len(manifest),
        "clean_incorrect": clean_incorrect,
        "source_invalid": source_invalid,
        "clean_correct_attackable": eligible,
        "successful_attacks": successes,
        "asr_on_clean_correct": successes / eligible if eligible else None,
        "robust_accuracy_on_clean_correct": 1.0 - successes / eligible if eligible else None,
        "robust_correct_fraction_all_processed": (
            (eligible - successes) / len(manifest) if manifest else None
        ),
    }


def evaluate(args):
    path_config.configure_paths(
        data_root=args.data_root,
        train_data_root=args.train_data_root,
        model_root=args.model_root,
        resource_root=args.resource_root,
        output_root=args.output_root,
        results_root=args.results_root,
    )
    root = output_dir(
        args.setting, args.attack, args.task, args.shift, args.attacked_model
    )
    config_file = root / "configuration.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"attack has not been run: {config_file}")
    configuration = json.loads(config_file.read_text(encoding="utf-8"))
    clean_rows = list(read_jsonl(root / "clean.jsonl"))
    adversarial_rows = list(read_jsonl(root / "adversarial.jsonl"))
    if len(clean_rows) != len(adversarial_rows):
        raise RuntimeError("clean/adversarial hard-set size mismatch")
    device = device_from_name(args.device)
    paths = model_paths(args.task, args.shift, args.evaluated_model)
    if args.task in {"code_classification", "defect_detection"}:
        labels = 250 if args.task == "code_classification" else 1
        model, tokenizer = load_classification(paths, device, num_labels=labels)
        victim = (ClassificationVictim if args.task == "code_classification" else BinaryClassificationVictim)(model, tokenizer, device)
        clean = classification_metrics(victim, clean_rows, args.batch_size)
        adversarial = classification_metrics(victim, adversarial_rows, args.batch_size)
        primary_clean, primary_adv = clean["accuracy"], adversarial["accuracy"]
    else:
        gallery_path = Path(configuration.get("gallery", configuration["source"]))
        source_rows = list(read_jsonl(gallery_path))
        model, tokenizer = load_retrieval(paths, device)
        victim = RetrievalVictim(
            model, tokenizer, device,
            [row["code"] for row in source_rows],
            [label_of(row) for row in source_rows],
            [str(row.get("submission_id", index)) for index, row in enumerate(source_rows)],
            batch_size=args.batch_size,
        )
        clean = retrieval_metrics(victim, clean_rows, args.batch_size)
        adversarial = retrieval_metrics(victim, adversarial_rows, args.batch_size)
        primary_clean, primary_adv = clean["p_at_1"], adversarial["p_at_1"]
    result = {
        "setting": args.setting,
        "attack": args.attack,
        "task": args.task,
        "shift": args.shift,
        "attacked_model": args.attacked_model,
        "evaluated_model": args.evaluated_model,
        "hard_sample_count": len(adversarial_rows),
        "clean_corresponding_hard_subset": clean,
        "adversarial_hard_subset": adversarial,
        "absolute_drop_on_hard_subset": (
            None if primary_clean is None else primary_clean - primary_adv
        ),
    }
    if args.setting == "adaptive" and args.attacked_model == args.evaluated_model:
        result["adaptive_all_attempts"] = adaptive_denominator(root)
    destination = (
        path_config.RESULTS_ROOT / args.setting / args.attack / args.task / args.shift
        / args.attacked_model / f"{args.evaluated_model}.json"
    )
    atomic_json(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("transfer", "adaptive"), required=True)
    parser.add_argument("--attack", choices=ATTACKS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--shift", choices=SHIFTS, required=True)
    parser.add_argument("--attacked-model", choices=MODELS, required=True)
    parser.add_argument("--evaluated-model", choices=MODELS, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--train-data-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    return parser.parse_args(argv)


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
