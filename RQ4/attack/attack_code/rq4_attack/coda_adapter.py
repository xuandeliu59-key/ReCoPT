from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fasttext
import numpy as np

from .data import normalized_sha256
from .itgen_adapter import AttackCandidate
from .java import JavaTools, valid_identifier
from .models import ClassificationVictim, RetrievalVictim


@dataclass
class Reference:
    label: int
    code: str
    variables: list[str]
    functions: list[str]
    vector_index: int


class CODAResources:
    def __init__(self, directory: Path):
        marker = directory / "_SUCCESS"
        if not marker.is_file():
            raise FileNotFoundError(
                f"CODA resources are incomplete: {directory}; run prepare_coda_resources.py"
            )
        self.references: list[Reference] = []
        with (directory / "references.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                self.references.append(Reference(**row))
        self.vectors = np.load(directory / "reference_vectors.npy", mmap_mode="r")
        if len(self.references) != len(self.vectors):
            raise RuntimeError("CODA reference metadata/vector count mismatch")
        self.fasttext = fasttext.load_model(str(directory / "identifiers.bin"))


def _cosine_rows(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    numerator = np.matmul(matrix, vector)
    denominator = np.linalg.norm(matrix, axis=1) * max(np.linalg.norm(vector), 1e-12)
    return numerator / np.maximum(denominator, 1e-12)


def _identifier_candidates(source: str, pool: list[str], model, limit: int) -> list[str]:
    source_vector = model.get_word_vector(source)
    candidates = []
    for name in dict.fromkeys(pool):
        if name == source or not valid_identifier(name):
            continue
        vector = model.get_word_vector(name)
        denominator = np.linalg.norm(source_vector) * np.linalg.norm(vector)
        similarity = float(np.dot(source_vector, vector) / max(denominator, 1e-12))
        candidates.append((similarity, name))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [name for _score, name in candidates[:limit]]


class CODAAttack:

    def __init__(
        self,
        victim,
        java: JavaTools,
        resources: CODAResources,
        reference_encoder,
        reference_tokenizer,
        query_budget=500,
        eval_batch_size=16,
        reference_count=64,
        candidates_per_identifier=50,
    ):
        self.victim = victim
        self.java = java
        self.resources = resources
        self.reference_encoder = reference_encoder
        self.reference_tokenizer = reference_tokenizer
        self.query_budget = int(query_budget)
        self.eval_batch_size = int(eval_batch_size)
        self.reference_count = int(reference_count)
        self.candidates_per_identifier = int(candidates_per_identifier)

    def _masked_vector(self, code: str, names: list[str]) -> np.ndarray:
        masked = self.java.replace_identifiers(code, {name: "unk" for name in names})
        return self.reference_encoder([masked], batch_size=1)[0]

    def _predict(self, codes: list[str], true_label: int):
        if isinstance(self.victim, ClassificationVictim):
            probabilities, predictions = self.victim.predict(codes, self.eval_batch_size)
            return probabilities, predictions, int(true_label)
        result = self.victim.predict(codes, self.eval_batch_size)
        return result.probabilities, result.predictions, 0

    def _select_references(self, vector: np.ndarray, true_label: int, clean_probs):
        eligible = np.ones(len(self.resources.references), dtype=bool)
        if isinstance(self.victim, ClassificationVictim):
            
            alternate = np.argsort(clean_probs[0])[::-1]
            alternate = [int(label) for label in alternate if int(label) != true_label]
            available = {reference.label for reference in self.resources.references}
            selected_labels = set(label for label in alternate if label in available)
            if selected_labels:
                eligible = np.asarray(
                    [reference.label in selected_labels for reference in self.resources.references]
                )
        indexes = np.flatnonzero(eligible)
        scores = _cosine_rows(np.asarray(self.resources.vectors[indexes]), vector)
        order = indexes[np.argsort(scores)[::-1][: self.reference_count]]
        return [self.resources.references[int(index)] for index in order]

    def attack(self, code: str, true_label: int) -> AttackCandidate:
        variables, functions = self.java.identifiers(code)
        names = variables + functions
        if not names:
            return AttackCandidate(code, False, 0.0, 0, {"reason": "no_identifiers"})
        query_start = self.victim.query
        clean_probs, clean_preds, target = self._predict([code], true_label)
        if int(clean_preds[0]) != target:
            return AttackCandidate(code, False, 0.0, 1, {"reason": "clean_incorrect"})
        clean_probability = float(clean_probs[0][target])

        masked_vector = self._masked_vector(code, names)
        references = self._select_references(masked_vector, true_label, clean_probs)
        variable_pool = [name for ref in references for name in ref.variables]
        function_pool = [name for ref in references for name in ref.functions]

        candidates: list[tuple[str, str, dict]] = []
        for style_name, styled in self.java.conservative_style_variants(code):
            candidates.append(("style", styled, {"style": style_name}))

        substitutions: dict[str, list[str]] = {}
        for name in variables:
            substitutions[name] = _identifier_candidates(
                name, variable_pool, self.resources.fasttext, self.candidates_per_identifier
            )
        for name in functions:
            substitutions[name] = _identifier_candidates(
                name, function_pool, self.resources.fasttext, self.candidates_per_identifier
            )
        max_rank = max((len(values) for values in substitutions.values()), default=0)
        for rank in range(max_rank):
            replacement = {}
            used = set(names)
            for name in names:
                options = substitutions.get(name, [])
                if rank < len(options) and options[rank] not in used:
                    replacement[name] = options[rank]
                    used.add(options[rank])
            if not replacement:
                continue
            renamed = self.java.replace_identifiers(code, replacement)
            if renamed != code and self.java.syntax_ok(renamed):
                candidates.append(("identifier", renamed, {"replacements": replacement, "rank": rank}))

        unique = []
        seen = {normalized_sha256(code)}
        for stage, candidate, metadata in candidates:
            digest = normalized_sha256(candidate)
            if digest not in seen:
                seen.add(digest)
                unique.append((stage, candidate, metadata))
        allowed = max(0, self.query_budget - (self.victim.query - query_start))
        unique = unique[:allowed]
        if not unique:
            return AttackCandidate(code, False, 0.0, self.victim.query - query_start, {"reason": "no_candidates"})

        best = None
        for offset in range(0, len(unique), self.eval_batch_size):
            batch = unique[offset:offset + self.eval_batch_size]
            probabilities, predictions, target = self._predict(
                [item[1] for item in batch], true_label
            )
            for item, probability, prediction in zip(batch, probabilities, predictions):
                drop = clean_probability - float(probability[target])
                record = (drop, item, int(prediction) != target)
                if best is None or drop > best[0]:
                    best = record
                if record[2]:
                    stage, adversarial, metadata = item
                    return AttackCandidate(
                        adversarial, True, drop, self.victim.query - query_start,
                        {
                            "stage": stage,
                            "reference_count": len(references),
                            "variables": variables,
                            "functions": functions,
                            "sha256": normalized_sha256(adversarial),
                            **metadata,
                        },
                    )
        drop, item, _success = best
        stage, final_code, metadata = item
        return AttackCandidate(
            final_code, False, float(drop), self.victim.query - query_start,
            {
                "reason": "search_failed",
                "best_stage": stage,
                "reference_count": len(references),
                **metadata,
            },
        )

