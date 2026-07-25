#!/bin/bash

set -euo pipefail

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation}"
: "${EXPERIMENT_CONFIG_RELATIVE:?Missing experiment config path}"
: "${RUN_PREFIX:?Missing result run prefix}"
: "${DATASET_LABEL:?Missing dataset label}"

JOB_STARTED_EPOCH=$(date +%s)

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

EXPERIMENT_PATH="${EXPERIMENT_PATH:-$HOME/_experiments/echollm-sage}"
SOURCE_PATH="${SOURCE_PATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL="${MODEL:-${DEFAULT_MODEL:-qwen3:4b-instruct}}"
RECORDED_LLM_PATH="${RECORDED_LLM_PATH:-}"
RECORD_OUTPUT_PATH="${RECORD_OUTPUT_PATH:-}"
PREPARED_PAIRS_PATH="${PREPARED_PAIRS_PATH:-}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -n "$ARRAY_TASK_ID" ]]; then
  TASK_NAME=$(printf 'task-%03d' "$ARRAY_TASK_ID")
  JOB_ROOT="$EXPERIMENT_PATH/jobs/$ARRAY_JOB_ID/$TASK_NAME"
  RESULT_GROUP="$RUN_PREFIX-$ARRAY_JOB_ID"
  RESULTS_OUTPUT_ROOT="$EXPERIMENT_PATH/results/$RESULT_GROUP/tasks"
  RUN_NAME="$TASK_NAME"
  RUN_SELECTION_ARGS=(--run-index "$ARRAY_TASK_ID" --skip-plots)
else
  JOB_ROOT="$EXPERIMENT_PATH/jobs/$SLURM_JOB_ID"
  RESULT_GROUP="$RUN_PREFIX-$SLURM_JOB_ID"
  RESULTS_OUTPUT_ROOT="$EXPERIMENT_PATH/results"
  RUN_NAME="$RESULT_GROUP"
  RUN_SELECTION_ARGS=()
fi
CODE_PATH="$JOB_ROOT/code"
LOG_PATH="$JOB_ROOT/logs"
HF_HOME="${HF_HOME:-$EXPERIMENT_PATH/huggingface-cache}"
EMBEDDING_CACHE="${EMBEDDING_CACHE_PATH:-$JOB_ROOT/embedding-cache.sqlite3}"
OLLAMA_LOG="$LOG_PATH/ollama.log"
PORT_SEED=$(printf '%s' "${ARRAY_JOB_ID}_${ARRAY_TASK_ID:-single}" | cksum | awk '{print $1}')
OLLAMA_PORT="${OLLAMA_PORT:-$((20000 + PORT_SEED % 40000))}"
OLLAMA_BIND="127.0.0.1:$OLLAMA_PORT"
OLLAMA_URL="http://$OLLAMA_BIND"

OLLAMA_PID=""

format_duration() {
  local total_seconds="$1"
  printf '%02d:%02d:%02d' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

stop_ollama() {
  if [[ -n "$OLLAMA_PID" ]]; then
    echo "Stopping Ollama ($OLLAMA_PID)..."
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
    OLLAMA_PID=""
  fi
}

cleanup() {
  local ec=$?
  trap - EXIT INT TERM
  stop_ollama
  if [[ -d "$CODE_PATH" && "$CODE_PATH" == "$JOB_ROOT/code" ]]; then
    rm -rf --one-file-system "$CODE_PATH"
  fi

  local elapsed status
  elapsed=$(($(date +%s) - JOB_STARTED_EPOCH))
  status="failed"
  if [[ "$ec" -eq 0 ]]; then
    status="completed"
  fi
  echo "Job $status after $(format_duration "$elapsed") ($elapsed seconds)."
  echo "Results: $RESULTS_OUTPUT_ROOT/$RUN_NAME"
  echo "Logs: $LOG_PATH"
  exit "$ec"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -f "$SOURCE_PATH/pyproject.toml" || ! -f "$SOURCE_PATH/_experiments/run.py" ]]; then
  echo "EchoLLM repository not found at SOURCE_PATH=$SOURCE_PATH" >&2
  echo "Submit the job from the repository root or set SOURCE_PATH explicitly." >&2
  exit 3
fi

mkdir -p \
  "$JOB_ROOT" \
  "$LOG_PATH" \
  "$RESULTS_OUTPUT_ROOT" \
  "$HF_HOME" \
  "$EXPERIMENT_PATH/locks"

echo "Copying a fresh working tree from $SOURCE_PATH..."
mkdir -p "$CODE_PATH"
rsync -a --delete \
  --exclude .git \
  --exclude .venv \
  --exclude .cache \
  --exclude .pytest_cache \
  --exclude __pycache__ \
  --exclude results \
  --exclude tmp \
  --exclude '*.out' \
  "$SOURCE_PATH/" "$CODE_PATH/"

CONFIG_PATH="$CODE_PATH/$EXPERIMENT_CONFIG_RELATIVE"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Experiment config not found: $CONFIG_PATH" >&2
  exit 3
fi

echo "Activating Python environment..."
eval "$(/storage/modules/packages/anaconda/bin/conda shell.bash hook)"
conda activate py313

if [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
  echo "Ensuring experiment requirements are installed..."
  exec 8>"$EXPERIMENT_PATH/locks/python-environment.lock"
  flock 8
  python -m pip install -q -r "$CODE_PATH/requirements-experiments.txt"
  flock -u 8
else
  echo "Using dependencies installed by the completed preparation job."
fi

export HF_HOME
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export OLLAMA_HOST="$OLLAMA_BIND"
export OLLAMA_GPU_LAYERS=9999
export OLLAMA_FLASH_ATTENTION=1

echo "Dataset: $DATASET_LABEL"
echo "Model: $MODEL"
echo "Experiment config: $CONFIG_PATH"
if [[ -n "$RECORDED_LLM_PATH" ]]; then
  if [[ ! -f "$RECORDED_LLM_PATH" ]]; then
    echo "Recorded LLM database not found: $RECORDED_LLM_PATH" >&2
    exit 3
  fi
  echo "Backend: recorded real responses from $RECORDED_LLM_PATH"
  LLM_ARGS=(--recorded-llm-path "$RECORDED_LLM_PATH")
else
  echo "Ollama endpoint: $OLLAMA_URL"
  echo "Starting Ollama..."
  # Keep startup/CUDA diagnostics while dropping high-volume access records.
  nohup ollama serve \
    > >(stdbuf -oL grep -v '^\[GIN\]' >>"$OLLAMA_LOG") \
    2>&1 &
  OLLAMA_PID=$!

  OLLAMA_READY=0
  for _ in {1..60}; do
    if curl -fs "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
      OLLAMA_READY=1
      break
    fi
    sleep 1
  done
  if [[ "$OLLAMA_READY" -ne 1 ]]; then
    echo "Ollama did not become ready at $OLLAMA_URL. Inspect $OLLAMA_LOG." >&2
    tail -n 30 "$OLLAMA_LOG" >&2 || true
    exit 2
  fi

  echo "Pulling model $MODEL (no-op when already available)..."
  exec 9>"$EXPERIMENT_PATH/locks/ollama-model.lock"
  flock 9
  ollama pull "$MODEL" >"$LOG_PATH/ollama_pull.log" 2>&1
  flock -u 9
  echo "Warming up Ollama on the allocated GPU..."
  echo "Warm up GPU" | ollama run "$MODEL" \
    >"$LOG_PATH/gpu-check.out" \
    2>>"$OLLAMA_LOG"

  if ! grep -Eq \
    'offload library=CUDA|offloaded [1-9][0-9]*/[0-9]+ layers to GPU|CUDA0' \
    "$OLLAMA_LOG"; then
    echo "Ollama did not report CUDA use. Inspect $OLLAMA_LOG." >&2
    exit 2
  fi
  LLM_ARGS=(--model "$MODEL" --ollama-host "$OLLAMA_URL")
fi

cd "$CODE_PATH"
if [[ "$EMBEDDING_DEVICE" == "cuda" ]]; then
  echo "Checking PyTorch CUDA access..."
  srun python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
fi

if [[ -n "$RECORD_OUTPUT_PATH" ]]; then
  echo "Recording real backend responses once for reuse by every policy task..."
  srun python -m _experiments.record \
    --config "$CONFIG_PATH" \
    --output "$RECORD_OUTPUT_PATH" \
    --prepared-pairs-output "$PREPARED_PAIRS_PATH" \
    --embedding-cache-path "$EMBEDDING_CACHE" \
    --device "$EMBEDDING_DEVICE" \
    --model "$MODEL" \
    --ollama-host "$OLLAMA_URL"
  echo "Recorded backend database: $RECORD_OUTPUT_PATH"
  exit 0
fi

if [[ -n "$ARRAY_TASK_ID" ]]; then
  echo "Running experiment grid index $ARRAY_TASK_ID..."
else
  echo "Running all policy/capacity experiments..."
fi
PREPARED_ARGS=()
if [[ -n "$PREPARED_PAIRS_PATH" ]]; then
  PREPARED_ARGS=(--prepared-pairs-path "$PREPARED_PAIRS_PATH")
fi
GRID_ARGS=()
if [[ -n "${POLICY_FILTER:-}" ]]; then
  GRID_ARGS+=(--policies "$POLICY_FILTER")
fi
if [[ "${POSITIVE_CACHE_SIZES_ONLY:-0}" == "1" ]]; then
  GRID_ARGS+=(--positive-cache-sizes-only)
fi
if [[ -n "${CACHE_SIZE_FILTER:-}" ]]; then
  IFS=',' read -r -a SELECTED_CACHE_SIZES <<<"$CACHE_SIZE_FILTER"
  GRID_ARGS+=(--cache-sizes "${SELECTED_CACHE_SIZES[@]}")
fi
srun python -m _experiments.run \
  --config "$CONFIG_PATH" \
  "${LLM_ARGS[@]}" \
  --device "$EMBEDDING_DEVICE" \
  --embedding-cache-path "$EMBEDDING_CACHE" \
  "${PREPARED_ARGS[@]}" \
  --output-dir "$RESULTS_OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  "${GRID_ARGS[@]}" \
  "${RUN_SELECTION_ARGS[@]}"

echo "Experiment completed successfully."
