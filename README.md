# ReCoPT Reproduction Package

This package contains the data, evaluation scripts, and results for the experimental evaluation of ReCoPT. The materials are organized by research question, and the reinforcement-learning components are included in a separate directory.

## Contents

- `RQ1/`: downstream evaluation data, scripts, and results.
- `RQ2/`: materials for representation-space and distribution-shift analyses.
- `RQ3/`: materials for the reward-design ablation study.
- `RQ4/`: robustness evaluation materials.
- `RL_repro/`: reinforcement-learning components.

## Model Checkpoints

Due to GitHub's file-size limits, the model checkpoints are hosted separately at:

```text
[RQ1](https://pan.quark.cn/s/dbb3ac682f78)
[RQ3](https://pan.quark.cn/s/d3ea6df65d74)
```

Download and extract the model checkpoints before running the evaluation. Keep the extracted directory structure unchanged so that each checkpoint can be matched with its corresponding task, shift, and method.

The pre-trained CodeBERT base model can be downloaded separately or loaded from an existing local directory.

## Usage

The following example evaluates the downloaded **Original** checkpoint on the CST ID split of the code-classification task.

Run the commands from the root of this reproduction package. Replace `BASE_MODEL` and `MODEL_ROOT` with the corresponding local directories.

```bash
ROOT="$(pwd)"
BASE_MODEL=/path/to/local/codebert-base
MODEL_ROOT=/path/to/extracted/model-archive

DATA="$ROOT/RQ1/package/data/stratified/cst/code_classification/id_test.jsonl"
CHECKPOINT="$MODEL_ROOT/RQ1/package/models/downstream/stratified/cst/code_classification/Original/checkpoint-best-acc/model.bin"
OUTPUT="$ROOT/test/rq1_original_classification_cst_id"

mkdir -p "$OUTPUT/checkpoint-best-acc"
ln -sfn "$CHECKPOINT" "$OUTPUT/checkpoint-best-acc/model.bin"

python "$ROOT/RQ1/package/scripts/tasks/code_classification/run.py" \
  --output_dir "$OUTPUT" \
  --model_type roberta \
  --model_name_or_path "$BASE_MODEL" \
  --tokenizer_name "$BASE_MODEL" \
  --do_test \
  --test_data_file "$DATA" \
  --eval_batch_size 16 \
  --block_size 512
```

`CHECKPOINT` is the downstream model evaluated in this example. `BASE_MODEL` supplies the CodeBERT architecture and tokenizer required to construct the model. Predictions are saved to:

```text
test/rq1_original_classification_cst_id/predictions.txt
```
