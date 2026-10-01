#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from classifier import DownstreamClassifierWrapper


ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = ROOT / "models/java250_codebert_classifier"

REFERENCE = ROOT / "data/reference/reference_10k.jsonl"
ID_DATA = ROOT / "data/reference/original_test_2k.jsonl"
METHODS = {
    "Random-View": (ROOT / "data/views/Random_View.jsonl", "downstream"),
    "LLM-View": (ROOT / "data/views/LLM_View.jsonl", "downstream"),
    "ContraBERT-View": (ROOT / "data/views/ContraBERT_View.jsonl", "downstream"),
    "ReCoPT-View": (ROOT / "data/views/ReCoPT_View.jsonl", "downstream"),
}
OUTPUT_DIR = ROOT / "results/ood"
METRICS_PATH = OUTPUT_DIR / "ood_metrics.json"
SCORES_PATH = OUTPUT_DIR / "ood_scores.npz"


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


def load_codes(path: Path, data_format: str) -> tuple[list[str], list[str]]:
    records = read_jsonl(path)
    ids = [record_id(record) for record in records]
    codes = [record_code(record, data_format) for record in records]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Missing or duplicate IDs in {path}")
    if not all(code.strip() for code in codes):
        raise ValueError(f"Empty code in {path}")
    return ids, codes


def problem_to_classifier_label() -> dict[str, int]:
    mapping: dict[str, int] = {}
    for record in read_jsonl(ID_DATA):
        problem = str(record.get("problem_id", record.get("label", "")))
        label = int(str(record["label"]).replace("label_", ""))
        previous = mapping.setdefault(problem, label)
        if previous != label:
            raise ValueError(f"Inconsistent label mapping for {problem}")
    if len(mapping) != 250 or set(mapping.values()) != set(range(250)):
        raise ValueError("Expected a bijection between 250 problems and labels 0..249")
    return mapping


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    auroc = roc_auc_score(labels, scores)
    aupr_in = average_precision_score(1 - labels, -scores)
    aupr_out = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = np.where(tpr >= 0.95)[0]
    tnr_at_95 = 1.0 - fpr[eligible[0]] if len(eligible) else np.nan
    return {
        "auroc": round(float(auroc), 4),
        "aupr_in": round(float(aupr_in), 4),
        "aupr_out": round(float(aupr_out), 4),
        "tnr_at_95": round(float(tnr_at_95), 4),
    }


def confidence_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    tensor = torch.from_numpy(logits)
    probabilities = functional.softmax(tensor, dim=1).numpy()
    return {
        "MSP": 1.0 - np.max(probabilities, axis=1),
        "Energy": (-torch.logsumexp(tensor, dim=1)).numpy(),
        "MaxLogit": -np.max(logits, axis=1),
    }


def mahalanobis_scores(
    features: np.ndarray,
    class_means: np.ndarray,
    precision: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    means_precision = class_means @ precision
    means_term = np.sum(means_precision * class_means, axis=1)
    result = []
    for start in range(0, len(features), batch_size):
        values = features[start : start + batch_size]
        values_precision = values @ precision
        values_term = np.sum(values_precision * values, axis=1, keepdims=True)
        distances = values_term - 2.0 * (values_precision @ class_means.T) + means_term
        result.append(np.min(distances, axis=1))
    return np.concatenate(result)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)

    mapping = problem_to_classifier_label()
    reference_records = read_jsonl(REFERENCE)
    reference_ids = [record_id(record) for record in reference_records]
    reference_codes = [record_code(record, "downstream") for record in reference_records]
    reference_labels = np.asarray([mapping[str(record["label"])] for record in reference_records])
    if len(reference_records) != 10000 or len(set(reference_ids)) != 10000:
        raise ValueError("Paper reference must contain 10,000 unique programs")
    label_counts = np.bincount(reference_labels, minlength=250)
    if np.any(label_counts == 0):
        raise ValueError("Paper reference does not cover every classifier label")

    id_ids, id_codes = load_codes(ID_DATA, "downstream")
    method_data: dict[str, tuple[list[str], list[str]]] = {}
    for method, (path, data_format) in METHODS.items():
        ids, codes = load_codes(path, data_format)
        if ids != id_ids:
            raise ValueError(f"{method} IDs/order do not match the common ID set")
        method_data[method] = (ids, codes)

    print(
        f"reference={len(reference_codes)} labels=250 "
        f"per_label={label_counts.min()}..{label_counts.max()} id={len(id_codes)}"
    )
    classifier = DownstreamClassifierWrapper(device=args.device)

    print("[1/6] Extracting reference features")
    reference_features, _ = classifier.get_features_and_logits(
        reference_codes, batch_size=args.batch_size, show_progress=True
    )

    print("[2/6] Building Mahalanobis and KNN references")
    class_means = np.stack(
        [np.mean(reference_features[reference_labels == label], axis=0) for label in range(250)]
    ).astype(np.float32)
    centered = reference_features - class_means[reference_labels]
    precision = EmpiricalCovariance().fit(centered).precision_.astype(np.float32)
    knn = NearestNeighbors(n_neighbors=args.k, metric="cosine", n_jobs=-1)
    knn.fit(reference_features)

    def score(codes: list[str], description: str) -> dict[str, np.ndarray]:
        print(description)
        features, logits = classifier.get_features_and_logits(
            codes, batch_size=args.batch_size, show_progress=True
        )
        scores = confidence_scores(logits)
        scores["Mahalanobis"] = mahalanobis_scores(features, class_means, precision)
        distances, _ = knn.kneighbors(features)
        scores["KNN"] = distances[:, -1]
        return scores

    id_scores = score(id_codes, "[3/6] Scoring ID programs")
    method_scores: dict[str, dict[str, np.ndarray]] = {}
    for index, (method, (_, codes)) in enumerate(method_data.items(), start=4):
        method_scores[method] = score(codes, f"[{index}/7] Scoring {method}")

    metric_order = ["MSP", "Energy", "MaxLogit", "Mahalanobis", "KNN"]
    output_metrics: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {
        f"ID__{metric}": values for metric, values in id_scores.items()
    }
    for method, scores in method_scores.items():
        output_metrics[method] = {
            metric: compute_ood_metrics(id_scores[metric], scores[metric])
            for metric in metric_order
        }
        output_metrics[method]["score_summary"] = {
            metric: summarize(scores[metric]) for metric in metric_order
        }
        for metric, values in scores.items():
            arrays[f"{method}__{metric}"] = values

    result = {
        "evaluation": "Paper/RQ2 model-aware OOD detection",
        "reference": str(REFERENCE.relative_to(ROOT)),
        "reference_selection": "seed-123 simple random sample without replacement after test-ID exclusion",
        "reference_count": len(reference_codes),
        "reference_labels": len(np.unique(reference_labels)),
        "reference_per_label_min": int(label_counts.min()),
        "reference_per_label_max": int(label_counts.max()),
        "id_data": str(ID_DATA.relative_to(ROOT)),
        "id_count": len(id_codes),
        "classifier_checkpoint": str((CLASSIFIER_DIR / "model.bin").relative_to(ROOT)),
        "classifier_base": str(CLASSIFIER_DIR.relative_to(ROOT)),
        "k": args.k,
        "score_direction": "higher means more OOD for every detector",
        "metrics": output_metrics,
        "id_score_summary": {
            metric: summarize(id_scores[metric]) for metric in metric_order
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(SCORES_PATH, **arrays)

    for method in METHODS:
        values = output_metrics[method]
        print(method, {metric: values[metric]["auroc"] for metric in metric_order})
    print(f"metrics={METRICS_PATH}")
    print(f"scores={SCORES_PATH}")


if __name__ == "__main__":
    main()
