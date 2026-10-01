#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/java250_codebert_classifier"


class CodeListDataset(Dataset):
    def __init__(self, codes: list[str], tokenizer, max_length: int = 512):
        self.codes = codes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            " ".join(self.codes[index].split()),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


class DownstreamClassifierWrapper:
    def __init__(self, device: str = "cuda", num_labels: int = 250):
        checkpoint = MODEL_DIR / "model.bin"
        required = [
            checkpoint,
            MODEL_DIR / "config.json",
            MODEL_DIR / "vocab.json",
            MODEL_DIR / "merges.txt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing packaged classifier files: {missing}")

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        config = RobertaConfig.from_pretrained(
            str(MODEL_DIR), num_labels=num_labels, local_files_only=True
        )
        self.model = RobertaForSequenceClassification(config)
        state_dict = torch.load(checkpoint, map_location="cpu")
        cleaned = {
            key[len("encoder.") :] if key.startswith("encoder.") else key: value
            for key, value in state_dict.items()
        }
        incompatible = self.model.load_state_dict(cleaned, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Classifier checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.model.to(self.device).eval()

    def _loader(self, codes: list[str], batch_size: int) -> DataLoader:
        return DataLoader(
            CodeListDataset(codes, self.tokenizer),
            batch_size=batch_size,
            shuffle=False,
        )

    @torch.no_grad()
    def get_logits(
        self, codes: list[str], batch_size: int = 8, show_progress: bool = False
    ) -> np.ndarray:
        loader = self._loader(codes, batch_size)
        if show_progress:
            from tqdm import tqdm
            loader = tqdm(loader, desc="Computing logits")
        batches = []
        for batch in loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            batches.append(self.model(**batch).logits.cpu().numpy())
        return np.concatenate(batches)

    @torch.no_grad()
    def get_features_and_logits(
        self, codes: list[str], batch_size: int = 8, show_progress: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        loader = self._loader(codes, batch_size)
        if show_progress:
            from tqdm import tqdm
            loader = tqdm(loader, desc="Extracting features+logits")
        feature_batches = []
        logit_batches = []
        for batch in loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            hidden = self.model.roberta(**batch).last_hidden_state
            feature_batches.append(hidden[:, 0, :].cpu().numpy())
            logit_batches.append(self.model.classifier(hidden).cpu().numpy())
        return np.concatenate(feature_batches), np.concatenate(logit_batches)
