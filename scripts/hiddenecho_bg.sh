#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/songhanlin/code/hidden-echo-paper-reproduction"
PYTHON_BIN="/data/songhanlin/envs/qwen_finetune/bin/python"

MODEL_PATH="${MODEL_PATH:-/data1/models/models--Qwen--Qwen2.5-1.5B-Instruct}"
EXP_NAME="${EXP_NAME:-hiddenecho_full_qwen25_20epoch}"
EPOCHS="${EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-48}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/songhanlin/tmp/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/songhanlin/tmp/hf-datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/songhanlin/tmp/hf-cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

SKIP_LAYERS="${SKIP_LAYERS:--1}"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
PID_FILE="$LOG_DIR/${EXP_NAME}.pid"

cd "$ROOT"

{
  echo "started_at: $(date -Is)"
  echo "model_path: $MODEL_PATH"
  echo "experiment_name: $EXP_NAME"
  echo "epochs: $EPOCHS"
  echo "train_batch_size: $TRAIN_BATCH_SIZE"
  echo "eval_batch_size: $EVAL_BATCH_SIZE"
  echo "skip_layers: $SKIP_LAYERS"
  SECONDS=0

  "$PYTHON_BIN" train_split.py \
    --experiment_name "$EXP_NAME" \
    --model_path "$MODEL_PATH" \
    --dataset_name financial_phrasebank \
    --num_train_epochs "$EPOCHS" \
    --lr_scheduler_type constant \
    --learning_rate 4e-4 \
    --max_len 128 \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --lora_rank 16 \
    --privacy_budget 5000 \
    --noise_type Chi \
    --lst_enable true \
    --lst_reduce_factor 16 \
    --lst_input_type clean \
    --lst_skip $SKIP_LAYERS \
    --lst_random_init false \
    --auto_skip false \
    --mi_downsample_enable false

  status=$?
  echo "finished_at: $(date -Is)"
  echo "exit_status: $status"
  echo "elapsed_seconds: $SECONDS"
  exit "$status"
} >"$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "started"
echo "pid: $(cat "$PID_FILE")"
echo "log: $LOG_FILE"
echo "outputs: $ROOT/outputs/train_ckpts/$EXP_NAME"
