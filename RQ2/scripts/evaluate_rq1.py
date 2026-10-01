#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache/embeddings"
DEFAULT_MODEL = "microsoft/codebert-base"

METHODS = {
    "Random_View": (DATA / "views/Random_View.jsonl", "downstream"),
    "LLM_View": (DATA / "views/LLM_View.jsonl", "downstream"),
    "ContraBERT_View": (DATA / "views/ContraBERT_View.jsonl", "downstream"),
    "ReCoPT_View": (DATA / "views/ReCoPT_View.jsonl", "downstream"),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_id(record: dict) -> str:
    return str(record.get("submission_id", record.get("id", record.get("idx", ""))))


def record_code(record: dict, data_format: str) -> str:
    if data_format == "contrastive":
        positives = record.get("positives") or []
        return str(positives[0]) if positives else str(record.get("anchor", ""))
    return str(record.get("code", record.get("func", "")))


class CodeDataset(Dataset):
    def __init__(self, codes: list[str], tokenizer, max_length: int):
        self.codes = codes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            " ".join(self.codes[index].split()),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


@torch.no_grad()
def encode(
    codes: list[str],
    destination: Path,
    model_path: str,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if destination.is_file() and not force:
        cached = np.load(destination, allow_pickle=False)
        if cached.shape == (len(codes), 768) and np.isfinite(cached).all():
            print(f"reuse={destination}")
            return cached
        raise ValueError(f"Invalid cached embedding: {destination} {cached.shape}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    loader = DataLoader(
        CodeDataset(codes, tokenizer, 512),
        batch_size=batch_size,
        shuffle=False,
    )
    batches = []
    for batch in tqdm(loader, desc=destination.stem):
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch).last_hidden_state[:, 0, :]
        batches.append(output.cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(batches)
    np.save(destination, embeddings)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.20)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)
    device = torch.device(args.device)
    CACHE.mkdir(parents=True, exist_ok=True)

    original_records = sorted(
        read_jsonl(DATA / "reference/original_test_2k.jsonl"), key=record_id
    )
    original_ids = [record_id(record) for record in original_records]
    if len(original_ids) != 2000 or len(set(original_ids)) != 2000:
        raise ValueError("Expected 2,000 unique source programs")
    original_codes = [record_code(record, "downstream") for record in original_records]
    encode(
        original_codes,
        CACHE / "codebert_Original.npy",
        args.model_name_or_path,
        device,
        args.batch_size,
        args.force,
    )

    for method, (path, data_format) in METHODS.items():
        records = sorted(read_jsonl(path), key=record_id)
        ids = [record_id(record) for record in records]
        if ids != original_ids:
            raise ValueError(f"ID/order mismatch for {method}")
        codes = [record_code(record, data_format) for record in records]
        if not all(code.strip() for code in codes):
            raise ValueError(f"Empty code in {method}")
        encode(
            codes,
            CACHE / f"codebert_{method}.npy",
            args.model_name_or_path,
            device,
            args.batch_size,
            args.force,
        )

    reference_records = read_jsonl(DATA / "reference/reference_10k.jsonl")
    reference_ids = np.asarray([record_id(record) for record in reference_records])
    if reference_ids.shape != (10000,) or len(set(reference_ids.tolist())) != 10000:
        raise ValueError("Expected 10,000 unique reference programs")
    reference_codes = [record_code(record, "downstream") for record in reference_records]
    encode(
        reference_codes,
        CACHE / "codebert_reference_10k.npy",
        args.model_name_or_path,
        device,
        args.batch_size,
        args.force,
    )
    np.save(CACHE / "codebert_reference_10k_ids.npy", reference_ids)
    print("CodeBERT embedding cache is ready")


if __name__ == "__main__":
    main()
