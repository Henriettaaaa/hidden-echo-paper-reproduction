#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/songhanlin/code/hidden-echo-paper-reproduction"
PYTHON_BIN="/data/songhanlin/envs/qwen_finetune/bin/python"

MODEL_PATH="${MODEL_PATH:-/data1/models/models--Qwen--Qwen2.5-1.5B-Instruct}"
EXP_NAME="${EXP_NAME:-smoke_he_1epoch}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/songhanlin/tmp/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/songhanlin/tmp/hf-datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/songhanlin/tmp/hf-cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

range() {
  local start=$1
  local end=$2
  local i=$start
  while ((i < end)); do
    printf '%s ' "$i"
    ((i += 1))
  done
}

exclude() {
  local array1=("$@")
  local array2=($(range 0 28))
  local diff=()

  for item in "${array2[@]}"; do
    if [[ ! " ${array1[*]} " =~ " ${item} " ]]; then
      diff+=("$item")
    fi
  done

  printf '%s ' "${diff[@]}"
}

SKIP_LAYERS="${SKIP_LAYERS:-$(exclude 0 7 14 21 27)}"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
PID_FILE="$LOG_DIR/${EXP_NAME}.pid"

cd "$ROOT"

nohup "$PYTHON_BIN" train_split.py \
  --experiment_name "$EXP_NAME" \
  --model_path "$MODEL_PATH" \
  --dataset_name financial_phrasebank \
  --num_train_epochs 1 \
  --lr_scheduler_type constant \
  --learning_rate 4e-4 \
  --max_len 128 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --lora_rank 16 \
  --privacy_budget 5000 \
  --lst_enable true \
  --lst_reduce_factor 16 \
  --lst_input_type clean \
  --lst_skip $SKIP_LAYERS \
  --lst_random_init false \
  --auto_skip false \
  --mi_downsample_enable false \
  >"$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "started"
echo "pid: $(cat "$PID_FILE")"
echo "log: $LOG_FILE"
echo "outputs: $ROOT/outputs/train_ckpts/$EXP_NAME"
