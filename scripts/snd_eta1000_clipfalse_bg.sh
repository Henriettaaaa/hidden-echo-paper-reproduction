#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/songhanlin/code/hidden-echo-paper-reproduction"
PYTHON_BIN="/data/songhanlin/envs/qwen_finetune/bin/python"

MODEL_PATH="${MODEL_PATH:-/data1/models/models--Qwen--Qwen2.5-1.5B-Instruct}"
EXP_NAME="${EXP_NAME:-snd_qwen25_linear_lr1p5e4_20epoch_allagree_eta1000_clipfalse}"
EPOCHS="${EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-48}"
GPU_ID="${GPU_ID:-3}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TMPDIR="${TMPDIR:-/data/songhanlin/tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/songhanlin/tmp/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/songhanlin/tmp/hf-datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/songhanlin/tmp/hf-cache}"
export SND_CLIP_EMBEDDING_L2=false
export SND_CACHE_DIR="${SND_CACHE_DIR:-/data/songhanlin/tmp/snd-cache}"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$SND_CACHE_DIR"

LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
PID_FILE="$LOG_DIR/${EXP_NAME}.pid"
OUTPUT_DIR="$ROOT/outputs/train_ckpts/$EXP_NAME"

cd "$ROOT"

if [[ -e "$LOG_FILE" || -e "$PID_FILE" || -d "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing experiment artifacts for $EXP_NAME" >&2
  exit 1
fi

{
  echo "started_at: $(date -Is)"
  echo "model_path: $MODEL_PATH"
  echo "experiment_name: $EXP_NAME"
  echo "privacy_budget: 1000"
  echo "clip_embedding_l2: false"
  echo "financial_phrasebank_config: sentences_allagree"
  echo "lr_scheduler_type: linear"
  echo "learning_rate: 1.5e-4"
  echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
  echo "tmpdir: $TMPDIR"
  echo "snd_cache_dir: $SND_CACHE_DIR"
  SECONDS=0

  if [[ ! -f "$SND_CACHE_DIR/mixed_datasets/0/chunk_0.pt" || ! -f "$SND_CACHE_DIR/mixed_datasets/1000/chunk_0.pt" ]]; then
    echo "snd_prepare_cache: generating mixed datasets for budgets 0 and 1000"
    "$PYTHON_BIN" -m baselines.snd.data \
      --model_type qwen2 \
      --model_name "$MODEL_PATH" \
      --privacy_budgets 0 1000
  else
    echo "snd_prepare_cache: reusing existing mixed dataset cache"
  fi

  if ! find "$ROOT/baselines/snd/denoise_model/1000" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
    echo "snd_denoise: training denoise checkpoint for eta=1000"
    "$PYTHON_BIN" -m baselines.snd.train_denoise \
      --model_type qwen2 \
      --model_name "$MODEL_PATH" \
      --privacy_budget 1000 \
      --num_train_epochs 1 \
      --learning_rate 1e-4 \
      --per_device_train_batch_size 2 \
      --per_device_eval_batch_size 8 \
      --lora_r 64 \
      --lr_scheduler_type constant
  else
    echo "snd_denoise: reusing existing denoise checkpoint"
  fi

  echo "snd_task: training task model"
  "$PYTHON_BIN" -m baselines.snd.train_task \
    --model_name "$MODEL_PATH" \
    --dataset_name financial_phrasebank \
    --financial_phrasebank_config sentences_allagree \
    --privacy_budget 1000 \
    --num_train_epochs "$EPOCHS" \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type linear \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --lora_r 16 \
    --output_dir "$OUTPUT_DIR"

  status=$?
  echo "finished_at: $(date -Is)"
  echo "exit_status: $status"
  echo "elapsed_seconds: $SECONDS"
  exit "$status"
} >"$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "started"
echo "pid: $(cat "$PID_FILE")"
echo "gpu_id: $GPU_ID"
echo "log: $LOG_FILE"
echo "outputs: $OUTPUT_DIR"
