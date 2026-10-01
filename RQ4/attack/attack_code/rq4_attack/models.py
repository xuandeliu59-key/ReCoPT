from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    RobertaConfig, RobertaForSequenceClassification, RobertaModel, RobertaTokenizer,
)

from .config import ModelPaths


class ClassificationNetwork(nn.Module):

    def __init__(self, encoder, dropout_probability: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, input_ids=None):
        return self.dropout(self.encoder(input_ids, attention_mask=input_ids.ne(1))[0])


class RetrievalNetwork(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        if len(outputs) > 1 and outputs[1] is not None:
            return outputs[1]
        return outputs[0][:, 0, :]


def tokenize_codes(tokenizer, codes: list[str], block_size: int) -> torch.Tensor:
    encoded = []
    for code in codes:
        tokens = tokenizer.tokenize(" ".join(code.split()))[: block_size - 2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids += [tokenizer.pad_token_id] * (block_size - len(ids))
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


def _load_state(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_classification(paths: ModelPaths, device: torch.device, num_labels: int = 250):
    config = RobertaConfig.from_pretrained(str(paths.pretrained), local_files_only=True)
    config.num_labels = int(num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(str(paths.pretrained), local_files_only=True)
    encoder = RobertaForSequenceClassification.from_pretrained(
        str(paths.pretrained), config=config, local_files_only=True
    )
    model = ClassificationNetwork(encoder, dropout_probability=0.0)
    model.load_state_dict(_load_state(paths.checkpoint), strict=True)
    model.to(device).eval()
    return model, tokenizer


def load_retrieval(paths: ModelPaths, device: torch.device):
    config = RobertaConfig.from_pretrained(str(paths.pretrained), local_files_only=True)
    tokenizer = RobertaTokenizer.from_pretrained(str(paths.pretrained), local_files_only=True)
    encoder = RobertaModel.from_pretrained(
        str(paths.pretrained), config=config, local_files_only=True
    )
    model = RetrievalNetwork(encoder)
    missing, unexpected = model.load_state_dict(_load_state(paths.checkpoint), strict=False)
    allowed_missing = {name for name in missing if name.startswith("encoder.pooler.")}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            f"retrieval checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.to(device).eval()
    return model, tokenizer


class ClassificationVictim:
    def __init__(self, model, tokenizer, device, block_size=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.query = 0

    def predict(self, codes: list[str], batch_size=16):
        return self.predict_ids(
            tokenize_codes(self.tokenizer, codes, self.block_size), batch_size
        )

    def predict_ids(self, ids: torch.Tensor, batch_size=16):
        self.query += len(ids)
        loader = DataLoader(ids, sampler=SequentialSampler(ids), batch_size=batch_size)
        probabilities = []
        self.model.eval()
        for batch in loader:
            with torch.no_grad():
                probabilities.append(
                    torch.softmax(self.model(batch.to(self.device)), -1).cpu()
                )
        values = torch.cat(probabilities).numpy()
        return values, values.argmax(-1)


class BinaryClassificationVictim(ClassificationVictim):

    def predict_ids(self, ids: torch.Tensor, batch_size=16):
        self.query += len(ids)
        loader = DataLoader(ids, sampler=SequentialSampler(ids), batch_size=batch_size)
        probabilities = []
        self.model.eval()
        for batch in loader:
            with torch.no_grad():
                logits = self.model(batch.to(self.device))
                positive = torch.sigmoid(logits[:, 0]).cpu()
                probabilities.append(torch.stack([1.0 - positive, positive], dim=1))
        values = torch.cat(probabilities).numpy()
        return values, (values[:, 1] > 0.5).astype(np.int64)


@dataclass
class RetrievalResult:
    probabilities: np.ndarray
    predictions: np.ndarray
    best_positive: np.ndarray
    best_negative: np.ndarray


class RetrievalVictim:

    def __init__(
        self, model, tokenizer, device, gallery_codes, gallery_labels, gallery_ids,
        batch_size=32, block_size=512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.gallery_labels = np.asarray(gallery_labels, dtype=np.int64)
        self.gallery_ids = list(gallery_ids)
        self.query = 0
        self.source_index = None
        self.source_label = None
        self.gallery_vectors = self.embed(gallery_codes, batch_size, count_query=False)

    def embed(self, codes: list[str], batch_size=32, count_query=True):
        return self.embed_ids(
            tokenize_codes(self.tokenizer, codes, self.block_size),
            batch_size=batch_size,
            count_query=count_query,
        )

    def embed_ids(self, ids: torch.Tensor, batch_size=32, count_query=True):
        if count_query:
            self.query += len(ids)
        loader = DataLoader(ids, sampler=SequentialSampler(ids), batch_size=batch_size)
        vectors = []
        self.model.eval()
        for batch in loader:
            with torch.no_grad():
                vectors.append(self.model(batch.to(self.device)).cpu())
        return torch.cat(vectors).numpy()

    def set_source(self, source_index: int, source_label: int) -> None:
        self.source_index = int(source_index)
        self.source_label = int(source_label)

    def predict(self, codes: list[str], batch_size=16):
        return self.predict_ids(
            tokenize_codes(self.tokenizer, codes, self.block_size), batch_size
        )

    def predict_ids(self, ids: torch.Tensor, batch_size=16):
        if self.source_label is None:
            raise RuntimeError("set_source must be called before retrieval queries")
        vectors = self.embed_ids(ids, batch_size=batch_size)
        scores = np.matmul(vectors, self.gallery_vectors.T)
        if self.source_index is not None:
            scores[:, self.source_index] = -1e30
        positive = self.gallery_labels == self.source_label
        negative = ~positive
        if not positive.any() or not negative.any():
            raise RuntimeError("retrieval gallery needs positive and negative labels")
        best_positive = scores[:, positive].max(axis=1)
        best_negative = scores[:, negative].max(axis=1)
        two_scores = np.stack([best_positive, best_negative], axis=1)
        two_scores -= two_scores.max(axis=1, keepdims=True)
        exponent = np.exp(two_scores)
        probabilities = exponent / exponent.sum(axis=1, keepdims=True)
        return RetrievalResult(
            probabilities, two_scores.argmax(axis=1), best_positive, best_negative
        )


class CodeSearchVictim:

    def __init__(self, model, tokenizer, device, block_size=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = int(block_size)
        self.query = 0
        self.query_vector = None
        self.negative_threshold = None

    def embed_ids(self, ids: torch.Tensor, batch_size=16, count_query=True):
        if count_query:
            self.query += len(ids)
        loader = DataLoader(ids, sampler=SequentialSampler(ids), batch_size=batch_size)
        vectors = []
        self.model.eval()
        for batch in loader:
            with torch.no_grad():
                vectors.append(self.model(batch.to(self.device)).cpu())
        return torch.cat(vectors).numpy()

    def embed(self, texts: list[str], batch_size=16, count_query=True):
        return self.embed_ids(
            tokenize_codes(self.tokenizer, texts, self.block_size),
            batch_size=batch_size,
            count_query=count_query,
        )

    def set_query(self, query_text: str, negative_threshold: float) -> None:
        self.query_vector = self.embed(
            [query_text], batch_size=1, count_query=False
        )[0]
        self.negative_threshold = float(negative_threshold)

    def predict(self, codes: list[str], batch_size=16):
        return self.predict_ids(
            tokenize_codes(self.tokenizer, codes, self.block_size), batch_size
        )

    def predict_ids(self, ids: torch.Tensor, batch_size=16):
        if self.query_vector is None or self.negative_threshold is None:
            raise RuntimeError("set_query must be called before code-search queries")
        vectors = self.embed_ids(ids, batch_size=batch_size)
        positive_scores = vectors @ self.query_vector
        negative_scores = np.full_like(positive_scores, self.negative_threshold)
        two_scores = np.stack([positive_scores, negative_scores], axis=1)
        shifted = two_scores - two_scores.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / exponent.sum(axis=1, keepdims=True)
        return RetrievalResult(
            probabilities=probabilities,
            predictions=two_scores.argmax(axis=1),
            best_positive=positive_scores,
            best_negative=negative_scores,
        )


class ITGenDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        item = self.features[index]
        return torch.tensor(item.input_ids), torch.tensor(item.label)


class ITGenVictimAdapter:

    def __init__(self, victim, tokenizer):
        self.victim = victim
        self.tokenizer = tokenizer
        self.query = victim.query

    def get_results(self, dataset, batch_size):
        ids = torch.stack([dataset[index][0] for index in range(len(dataset))])
        before = self.victim.query
        if isinstance(self.victim, ClassificationVictim):
            probabilities, predictions = self.victim.predict_ids(ids, batch_size)
        else:
            result = self.victim.predict_ids(ids, batch_size)
            probabilities, predictions = result.probabilities, result.predictions
        self.query += self.victim.query - before
        return probabilities, predictions


def device_from_name(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)

