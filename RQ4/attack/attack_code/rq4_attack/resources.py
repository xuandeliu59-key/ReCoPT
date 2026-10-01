from __future__ import annotations

import json
import random
from pathlib import Path

import fasttext
import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from . import config
from .data import atomic_json, atomic_text, label_of, read_jsonl
from .java import JavaTools
from .models import tokenize_codes


class ReferenceEncoder:

    def __init__(self, device: torch.device, block_size=512):
        pretrained = config.require_model_root() / "pretrained/shared/Original"
        model_config = RobertaConfig.from_pretrained(str(pretrained), local_files_only=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(
            str(pretrained), local_files_only=True
        )
        self.model = RobertaModel.from_pretrained(
            str(pretrained), config=model_config, local_files_only=True
        ).to(device).eval()
        self.device = device
        self.block_size = int(block_size)

    def __call__(self, codes: list[str], batch_size=16) -> np.ndarray:
        vectors = []
        for offset in range(0, len(codes), batch_size):
            ids = tokenize_codes(
                self.tokenizer, codes[offset:offset + batch_size], self.block_size
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(ids, attention_mask=ids.ne(1))
                vector = outputs[1] if outputs[1] is not None else outputs[0][:, 0, :]
            vectors.append(vector.cpu().numpy())
        return np.concatenate(vectors, axis=0)


def _reservoir_by_label(rows, maximum: int, seed: int):
    rng = random.Random(seed)
    groups: dict[int, list[dict]] = {}
    counts: dict[int, int] = {}
    for row in rows:
        label = label_of(row)
        counts[label] = counts.get(label, 0) + 1
        bucket = groups.setdefault(label, [])
        if len(bucket) < maximum:
            bucket.append(row)
        else:
            index = rng.randrange(counts[label])
            if index < maximum:
                bucket[index] = row
    return groups


def prepare_coda_resources(
    train_path: Path,
    output_dir: Path,
    device: torch.device,
    max_refs_per_label=16,
    seed=123456,
    batch_size=16,
    fasttext_threads=8,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    java = JavaTools()
    groups = _reservoir_by_label(read_jsonl(train_path), max_refs_per_label, seed)
    selected = []
    corpus_lines = []
    for label in sorted(groups):
        for row in groups[label]:
            code = row["code"]
            variables, functions = java.identifiers(code)
            if not variables and not functions:
                continue
            selected.append((label, code, variables, functions))
            corpus_lines.append(" ".join(variables + functions))
    if not selected:
        raise RuntimeError("no Java references with identifiers were found")

    corpus_path = output_dir / "identifier_corpus.txt"
    atomic_text(corpus_path, "\n".join(corpus_lines) + "\n")
    ft_model = fasttext.train_unsupervised(
        str(corpus_path), model="skipgram", dim=100, epoch=5, minCount=1,
        minn=2, maxn=6, thread=max(1, int(fasttext_threads)), verbose=0,
    )
    ft_model.save_model(str(output_dir / "identifiers.bin"))

    encoder = ReferenceEncoder(device)
    masked_codes = [
        java.replace_identifiers(code, {name: "unk" for name in variables + functions})
        for _label, code, variables, functions in selected
    ]
    vectors = encoder(masked_codes, batch_size=batch_size).astype(np.float16)
    np.save(output_dir / "reference_vectors.npy", vectors)

    lines = []
    for index, (label, code, variables, functions) in enumerate(selected):
        lines.append(json.dumps({
            "label": label,
            "code": code,
            "variables": variables,
            "functions": functions,
            "vector_index": index,
        }, ensure_ascii=False))
    atomic_text(output_dir / "references.jsonl", "\n".join(lines) + "\n")
    summary = {
        "source": str(train_path),
        "reference_count": len(selected),
        "label_count": len({item[0] for item in selected}),
        "max_refs_per_label": int(max_refs_per_label),
        "seed": int(seed),
        "embedding": "CodeBERT pooled masked-code vector, float16",
        "identifier_model": "FastText skipgram dim=100 minn=2 maxn=6",
        "language": "java",
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_text(output_dir / "_SUCCESS", "complete\n")
    return summary

