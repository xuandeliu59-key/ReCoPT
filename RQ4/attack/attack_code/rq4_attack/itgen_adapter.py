from __future__ import annotations

import copy
import importlib.util
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from botorch.models.gp_regression import SingleTaskGP

from .data import normalized_sha256
from .java import JavaTools, valid_identifier
from .models import ITGenDataset, ITGenVictimAdapter


ATTACK_CODE_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(
    os.environ.get("RQ4_ITGEN_ROOT", str(ATTACK_CODE_ROOT / "ITGen_official"))
).expanduser().resolve()


def configure_itgen_root(path: Path | str | None) -> None:
    global OFFICIAL
    if path is not None:
        OFFICIAL = Path(path).expanduser().resolve()


@dataclass
class ITGenFeature:
    input_tokens: list[str]
    input_ids: list[int]
    label: int


@dataclass
class AttackCandidate:
    code: str
    success: bool
    score_drop: float
    queries: int
    metadata: dict


def _install_botorch_compatibility() -> None:
    import botorch.utils.containers as containers

    if not hasattr(containers, "TrainingData"):
        class TrainingData:
            def __init__(self, X, Y):
                self.X = X
                self.Y = Y
        containers.TrainingData = TrainingData
    if not hasattr(SingleTaskGP, "get_batch_dimensions"):
        SingleTaskGP.get_batch_dimensions = staticmethod(
            lambda train_X, train_Y: (train_X.shape[:-2], train_X.shape[:-2])
        )


def _stable_update_queue(block_queue, _index_dict):
    return sorted(block_queue, key=lambda item: (item[0], item[1]))


def load_official_itgen(java: JavaTools):
    _install_botorch_compatibility()
    for directory in (OFFICIAL, OFFICIAL / "algorithms", OFFICIAL / "attack"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    parser_package = types.ModuleType("python_parser")
    parser_package.__path__ = []
    parser_module = types.ModuleType("python_parser.run_parser")

    def identifiers(code, _lang="java"):
        variables, _functions = java.identifiers(code)
        return [[name] for name in variables], java.token_texts(code)

    def replace_batch(code, replacements, _lang="java"):
        return java.replace_identifiers(code, replacements)

    def generated(code, new_tokens, _lang="java"):
        old_tokens = java.token_texts(code)
        replacements = {
            old: new for old, new in zip(old_tokens, new_tokens)
            if old != new and valid_identifier(old) and valid_identifier(new)
        }
        return java.replace_identifiers(code, replacements)

    parser_module.get_identifiers = identifiers
    parser_module.get_example_batch = replace_batch
    parser_module.get_example = lambda code, old, new, _lang="java": replace_batch(
        code, {old: new}
    )
    parser_module.get_gen_code = generated
    parser_package.run_parser = parser_module
    sys.modules["python_parser"] = parser_package
    sys.modules["python_parser.run_parser"] = parser_module
    sys.modules["run_parser"] = parser_module

    utils_spec = importlib.util.spec_from_file_location("utils", OFFICIAL / "utils.py")
    if utils_spec is None or utils_spec.loader is None:
        raise RuntimeError("cannot load vendored ITGen utils.py")
    official_utils = importlib.util.module_from_spec(utils_spec)
    sys.modules["utils"] = official_utils
    utils_spec.loader.exec_module(official_utils)

    source_path = OFFICIAL / "attack/ITGenAttacker.py"
    source = source_path.read_text(encoding="utf-8")
    
    
    
    duplicated = '''        while self.BLOCK_QUEUE:\n            stage_call, fX, X, fidx = self.exploration_ball_with_indices(center_seq=center_seq,n_samples=n_samples,ball_size=ex_ball_size,stage_call=stage_call, opt_indices=opt_indices, KEY=KEY, stage_init_ind=stage_init_ind)\n\n            if stage_call == -1:\n                new_code_tokens = self.seq2code(X)\n                is_success = self.is_success(new_code_tokens)\n                if is_success == 1:\n                    return self.adv_code(code_1, new_code_tokens), is_success, self.replaced_words(self.code_tokens_1, new_code_tokens)\n                else:\n                    return None, 0, None\n        \n'''
    if source.count(duplicated) != 1:
        raise RuntimeError("unexpected ITGen release; compatibility block no longer matches")
    source = source.replace(duplicated, "", 1)
    module = types.ModuleType("rq4_itgen_official_compat")
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)

    module.CodeDataset = ITGenDataset
    module.get_identifiers = identifiers
    module.get_example_batch = replace_batch
    module.get_gen_code = generated
    module.ITGen_Attacker.update_queue = staticmethod(_stable_update_queue)

    def convert_features(code1_tokens, _code2, label, _u1, _u2, tokenizer, args, _cache):
        tokens = code1_tokens[: args.block_size - 2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids += [tokenizer.pad_token_id] * (args.block_size - len(ids))
        return ITGenFeature(tokens, ids, int(label))

    module.convert_examples_to_features = convert_features
    return module, official_utils


def build_identifier_vocab(rows, java: JavaTools, limit=10000) -> list[str]:
    counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        if index >= limit:
            break
        variables, functions = java.identifiers(row["code"])
        for name in variables + functions:
            if valid_identifier(name):
                counts[name] = counts.get(name, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _substitutions(names, vocabulary, code, candidates_per_variable, rng):
    available = [name for name in vocabulary if name not in names and name not in code]
    if not available:
        return {}
    result = {}
    for name in names:
        pool = available if len(available) <= candidates_per_variable else rng.sample(
            available, candidates_per_variable
        )
        result[name] = list(pool)
    return result


class ITGenAttack:
    def __init__(
        self,
        victim,
        tokenizer,
        java: JavaTools,
        vocabulary: list[str],
        query_budget=500,
        time_limit=120,
        candidates_per_variable=50,
        eval_batch_size=16,
        seed=123456,
    ):
        self.victim = victim
        self.tokenizer = tokenizer
        self.java = java
        self.vocabulary = vocabulary
        self.query_budget = int(query_budget)
        self.time_limit = int(time_limit)
        self.candidates_per_variable = int(candidates_per_variable)
        self.eval_batch_size = int(eval_batch_size)
        self.rng = random.Random(seed)
        self.module, self.official_utils = load_official_itgen(java)
        self.adapter = ITGenVictimAdapter(victim, tokenizer)
        self.args = SimpleNamespace(
            block_size=512,
            eval_batch_size=self.eval_batch_size,
            dropout_probability=0.0,
            query_budget=self.query_budget,
            time_limit_seconds=self.time_limit,
        )
        official_class = self.module.ITGen_Attacker
        original_budget = official_class.get_query_budget
        official_class.get_query_budget = (
            lambda attacker, vertices: min(
                original_budget(attacker, vertices), attacker.args.query_budget
            )
        )
        official_class.check_query_const = lambda attacker: (
            attacker.model.query - attacker.query_times >= attacker.query_budget
            or time.time() - attacker.start_time >= attacker.args.time_limit_seconds
        )

    def _feature(self, code: str, label: int):
        tokens = self.tokenizer.tokenize(" ".join(code.split()))
        return self.module.convert_examples_to_features(
            tokens, [], label, None, None, self.tokenizer, self.args, None
        )

    def attack(self, code: str, true_label: int) -> AttackCandidate:
        variables, _functions = self.java.identifiers(code)
        if not variables:
            return AttackCandidate(code, False, 0.0, 0, {"reason": "no_variables"})
        per_variable = max(
            1,
            min(
                self.candidates_per_variable,
                self.query_budget // max(2 * len(variables), 1),
            ),
        )
        substitutions = _substitutions(
            variables, self.vocabulary, code, per_variable, self.rng
        )
        if not substitutions:
            return AttackCandidate(code, False, 0.0, 0, {"reason": "no_substitutions"})

        attack_label = int(true_label)
        if self.victim.__class__.__name__ in {"RetrievalVictim", "CodeSearchVictim"}:
            attack_label = 0
        feature = self._feature(code, attack_label)
        dataset = ITGenDataset([feature])
        original_probs, original_preds = self.adapter.get_results(dataset, 1)
        if int(original_preds[0]) != attack_label:
            return AttackCandidate(code, False, 0.0, 1, {"reason": "clean_incorrect"})
        query_start = self.adapter.query
        attacker = self.module.ITGen_Attacker(self.args, self.adapter, self.tokenizer)
        adversarial, success, replacements = attacker.itgen_attack(
            dataset[0], copy.deepcopy(substitutions), (None, None, code, ""),
            query_start, original_probs, time.time()
        )
        queries = self.adapter.query - query_start
        if not success or not adversarial or not self.java.syntax_ok(adversarial):
            return AttackCandidate(
                code, False, 0.0, queries,
                {"reason": "search_failed"}
            )
        final_feature = self._feature(adversarial, attack_label)
        final_probs, final_preds = self.adapter.get_results(ITGenDataset([final_feature]), 1)
        successful = int(final_preds[0]) != attack_label
        drop = float(original_probs[0][attack_label] - final_probs[0][attack_label])
        return AttackCandidate(
            adversarial,
            successful,
            drop,
            queries + 1,
            {
                "replacements": replacements or {},
                "candidate_count_per_variable": per_variable,
                "variables": variables,
                "sha256": normalized_sha256(adversarial),
            },
        )

