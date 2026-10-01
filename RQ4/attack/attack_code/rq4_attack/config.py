from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ATTACK_CODE_ROOT = Path(__file__).resolve().parents[1]
BUDGET_ROOT = ATTACK_CODE_ROOT.parent
EXPERIMENT_ROOT = BUDGET_ROOT.parent


def _configured_path(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve() if default is not None else None


DATA_ROOT = _configured_path("RQ4_DATA_ROOT", EXPERIMENT_ROOT / "data")
TRAIN_DATA_ROOT = _configured_path("RQ4_TRAIN_DATA_ROOT", DATA_ROOT)
MODEL_ROOT = _configured_path("RQ4_MODEL_ROOT")
RESOURCE_ROOT = _configured_path("RQ4_RESOURCE_ROOT", EXPERIMENT_ROOT / "resources")
OUTPUT_ROOT = _configured_path("RQ4_OUTPUT_ROOT", BUDGET_ROOT / "generated")
RESULTS_ROOT = _configured_path("RQ4_RESULTS_ROOT", BUDGET_ROOT / "results")


def configure_paths(
    *,
    data_root: Path | str | None = None,
    train_data_root: Path | str | None = None,
    model_root: Path | str | None = None,
    resource_root: Path | str | None = None,
    output_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> None:
    global DATA_ROOT, TRAIN_DATA_ROOT, MODEL_ROOT, RESOURCE_ROOT, OUTPUT_ROOT, RESULTS_ROOT

    def resolved(value: Path | str | None, current: Path | None) -> Path | None:
        return Path(value).expanduser().resolve() if value is not None else current

    DATA_ROOT = resolved(data_root, DATA_ROOT)
    TRAIN_DATA_ROOT = resolved(train_data_root, TRAIN_DATA_ROOT)
    MODEL_ROOT = resolved(model_root, MODEL_ROOT)
    RESOURCE_ROOT = resolved(resource_root, RESOURCE_ROOT)
    OUTPUT_ROOT = resolved(output_root, OUTPUT_ROOT)
    RESULTS_ROOT = resolved(results_root, RESULTS_ROOT)


def require_model_root() -> Path:
    if MODEL_ROOT is None:
        raise ValueError("model root is required; pass --model-root or set RQ4_MODEL_ROOT")
    return MODEL_ROOT
SHIFTS = ("cst", "token")
TASKS = (
    "code_classification", "clone_detection", "defect_detection",
)
MODELS = ("Original", "ContraBERT_C", "ReCoPT")
ATTACKS = ("itgen", "coda")


@dataclass(frozen=True)
class ModelPaths:
    pretrained: Path
    checkpoint: Path


def split_for_task(task: str) -> str:
    if task in {"code_classification", "defect_detection"}:
        return "stratified"
    if task == "clone_detection":
        return "label"
    raise ValueError(f"unsupported task: {task}")


def data_path(task: str, shift: str, role: str) -> Path:
    _validate(task, shift)
    if role not in {"train", "id_test", "ood_test"}:
        raise ValueError(f"unsupported role: {role}")
    if role in {"id_test", "ood_test"}:
        return DATA_ROOT / split_for_task(task) / shift / task / f"{role}.jsonl"
    return TRAIN_DATA_ROOT / task / shift / "train.jsonl"


def model_paths(task: str, shift: str, model: str) -> ModelPaths:
    _validate(task, shift)
    if model not in MODELS:
        raise ValueError(f"unsupported model: {model}")
    model_root = require_model_root()
    split = split_for_task(task)
    if model == "Original":
        pretrained = model_root / "pretrained/shared/Original"
    elif model == "ContraBERT_C":
        pretrained = model_root / "pretrained/shared/ContraBERT_C"
    else:
        pretrained = model_root / "pretrained" / split / shift / "ReCoPT"
    checkpoint_name = {
        "code_classification": "checkpoint-best-acc",
        "defect_detection": "checkpoint-best-acc",
        "clone_detection": "checkpoint-best-map",
    }[task]
    checkpoint = (
        model_root / "downstream" / split / shift / task / model
        / checkpoint_name / "model.bin"
    )
    return ModelPaths(pretrained=pretrained, checkpoint=checkpoint)


def resource_dir(task: str, shift: str) -> Path:
    _validate(task, shift)
    return RESOURCE_ROOT / task / shift / "coda"


def output_dir(setting: str, attack: str, task: str, shift: str, attacked_model: str) -> Path:
    if setting not in {"transfer", "adaptive"}:
        raise ValueError(f"unsupported setting: {setting}")
    if attack not in ATTACKS:
        raise ValueError(f"unsupported attack: {attack}")
    _validate(task, shift)
    if attacked_model not in MODELS:
        raise ValueError(f"unsupported model: {attacked_model}")
    return OUTPUT_ROOT / setting / attack / task / shift / attacked_model


def validate_paths(task: str, shift: str, model: str) -> None:
    paths = model_paths(task, shift, model)
    required = [
        data_path(task, shift, "train"),
        data_path(task, shift, "id_test"),
        paths.pretrained / "config.json",
        paths.pretrained / "vocab.json",
        paths.pretrained / "merges.txt",
        paths.checkpoint,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required RQ1 files:\n" + "\n".join(missing))


def _validate(task: str, shift: str) -> None:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    if shift not in SHIFTS:
        raise ValueError(f"unsupported shift: {shift}")
