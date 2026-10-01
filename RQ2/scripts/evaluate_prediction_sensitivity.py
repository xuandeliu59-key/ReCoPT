#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from classifier import DownstreamClassifierWrapper


ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = ROOT / "models/java250_codebert_classifier"
ID_DATA = ROOT / "data/reference/original_test_2k.jsonl"
METHODS = {
    "Random-View": (ROOT / "data/views/Random_View.jsonl", "downstream"),
    "LLM-View": (ROOT / "data/views/LLM_View.jsonl", "downstream"),
    "ContraBERT-View": (ROOT / "data/views/ContraBERT_View.jsonl", "downstream"),
    "ReCoPT-View": (ROOT / "data/views/ReCoPT_View.jsonl", "downstream"),
}
OUTPUT_DIR = ROOT / "results/prediction"
METRICS_PATH = OUTPUT_DIR / "prediction_metrics.json"
CSV_PATH = OUTPUT_DIR / "prediction_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_id(record: dict) -> str:
    return str(record.get("submission_id", record.get("id", record.get("idx", ""))))


def record_code(record: dict, data_format: str) -> str:
    if data_format == "downstream":
        return str(record.get("code", record.get("func", "")))
    positives = record.get("positives") or []
    return str(positives[0]) if positives else str(record.get("anchor", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.15)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)

    original_records = read_jsonl(ID_DATA)
    original_ids = [record_id(record) for record in original_records]
    original_codes = [record_code(record, "downstream") for record in original_records]
    original_labels = np.asarray([
        int(str(record["label"]).replace("label_", ""))
        for record in original_records
    ])
    if len(original_ids) != 2000 or len(set(original_ids)) != 2000:
        raise ValueError("Expected 2,000 unique original sample IDs")

    method_codes: dict[str, list[str]] = {}
    for method, (path, data_format) in METHODS.items():
        records = read_jsonl(path)
        by_id = {record_id(record): record_code(record, data_format) for record in records}
        if len(by_id) != 2000 or set(by_id) != set(original_ids):
            raise ValueError(f"{method} does not match the common 2,000-ID set")
        method_codes[method] = [by_id[sample_id] for sample_id in original_ids]
        if not all(code.strip() for code in method_codes[method]):
            raise ValueError(f"{method} contains empty code")

    print("aligned_samples=2000 methods=4")
    classifier = DownstreamClassifierWrapper(device=args.device)

    def predict(codes: list[str], description: str) -> np.ndarray:
        print(description)
        logits = classifier.get_logits(
            codes, batch_size=args.batch_size, show_progress=True
        )
        return np.argmax(logits, axis=1)

    original_predictions = predict(original_codes, "[1/5] Original")
    original_correct = original_predictions == original_labels
    original_accuracy = float(np.mean(original_correct))

    predictions: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, float | int]] = {}
    for index, method in enumerate(METHODS, start=2):
        shifted_predictions = predict(method_codes[method], f"[{index}/5] {method}")
        predictions[method] = shifted_predictions
        shifted_correct = shifted_predictions == original_labels
        shifted_accuracy = float(np.mean(shifted_correct))
        accuracy_drop = original_accuracy - shifted_accuracy
        flip_rate = float(np.mean(original_predictions != shifted_predictions))
        results[method] = {
            "sample_count": len(original_ids),
            "original_accuracy": original_accuracy,
            "shifted_accuracy": shifted_accuracy,
            "accuracy_drop": accuracy_drop,
            "relative_accuracy_drop": accuracy_drop / original_accuracy,
            "flip_rate": flip_rate,
            "original_correct_shifted_wrong": int(np.sum(original_correct & ~shifted_correct)),
            "original_wrong_shifted_correct": int(np.sum(~original_correct & shifted_correct)),
        }

    output = {
        "evaluation": "Paper/RQ2 downstream behavioral impact",
        "id_data": str(ID_DATA.relative_to(ROOT)),
        "sample_count": len(original_ids),
        "classifier_checkpoint": str((CLASSIFIER_DIR / "model.bin").relative_to(ROOT)),
        "classifier_base": str(CLASSIFIER_DIR.relative_to(ROOT)),
        "definitions": {
            "accuracy_drop": "original_accuracy - shifted_accuracy",
            "relative_accuracy_drop": "accuracy_drop / original_accuracy",
            "flip_rate": "mean(original_prediction != shifted_prediction)",
        },
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "Method", "Sample_Count", "Original_Accuracy", "Shifted_Accuracy",
            "Accuracy_Drop", "Relative_Accuracy_Drop", "Flip_Rate",
            "Original_Correct_Shifted_Wrong", "Original_Wrong_Shifted_Correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in results.items():
            writer.writerow({
                "Method": method,
                "Sample_Count": metrics["sample_count"],
                "Original_Accuracy": f'{metrics["original_accuracy"]:.6f}',
                "Shifted_Accuracy": f'{metrics["shifted_accuracy"]:.6f}',
                "Accuracy_Drop": f'{metrics["accuracy_drop"]:.6f}',
                "Relative_Accuracy_Drop": f'{metrics["relative_accuracy_drop"]:.6f}',
                "Flip_Rate": f'{metrics["flip_rate"]:.6f}',
                "Original_Correct_Shifted_Wrong": metrics["original_correct_shifted_wrong"],
                "Original_Wrong_Shifted_Correct": metrics["original_wrong_shifted_correct"],
            })

    with PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(original_ids):
            row = {
                "submission_id": sample_id,
                "label": int(original_labels[index]),
                "original_prediction": int(original_predictions[index]),
                "original_correct": bool(original_correct[index]),
                "methods": {},
            }
            for method in METHODS:
                prediction = int(predictions[method][index])
                row["methods"][method] = {
                    "prediction": prediction,
                    "correct": prediction == int(original_labels[index]),
                    "flipped": prediction != int(original_predictions[index]),
                }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for method, metrics in results.items():
        print(method, {
            key: round(float(metrics[key]), 4)
            for key in ("original_accuracy", "shifted_accuracy", "accuracy_drop", "flip_rate")
        })
    print(f"metrics={METRICS_PATH}")
    print(f"csv={CSV_PATH}")
    print(f"predictions={PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
