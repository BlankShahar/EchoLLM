#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=7-00:00:00
#SBATCH --job-name=echollm-sage-oasst1
#SBATCH --output=echollm-sage-oasst1-%j.out
#SBATCH --mem=58G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_6000:1

set -euo pipefail

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

EXPERIMENT_PATH="${EXPERIMENT_PATH:-/home/blanksh/_experiments/echollm-sage}"
SOURCE_PATH="${SOURCE_PATH:-/home/blanksh/EchoLLM}"
MODEL="${MODEL:-llama3.2:1b}"

JOB_ROOT="$EXPERIMENT_PATH/jobs/$SLURM_JOB_ID"
CODE_PATH="$JOB_ROOT/code"
LOG_PATH="$JOB_ROOT/logs"
RESULTS_ROOT="$EXPERIMENT_PATH/results"
RUN_NAME="oasst1-sage-$SLURM_JOB_ID"
HF_HOME="${HF_HOME:-$EXPERIMENT_PATH/huggingface-cache}"
EMBEDDING_CACHE="$EXPERIMENT_PATH/embedding-cache/echollm_embeddings.sqlite3"
OLLAMA_LOG="$LOG_PATH/ollama.log"

OLLAMA_PID=""

stop_ollama() {
  if [[ -n "$OLLAMA_PID" ]]; then
    echo "Stopping Ollama ($OLLAMA_PID)..."
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
    OLLAMA_PID=""
  fi
}

cleanup() {
  ec=$?
  trap - EXIT INT TERM
  stop_ollama
  if [[ -d "$CODE_PATH" && "$CODE_PATH" == "$JOB_ROOT/code" ]]; then
    rm -rf --one-file-system "$CODE_PATH"
  fi
  echo "Results: $RESULTS_ROOT/$RUN_NAME"
  echo "Logs: $LOG_PATH"
  exit "$ec"
}
trap cleanup EXIT INT TERM

mkdir -p \
  "$JOB_ROOT" \
  "$LOG_PATH" \
  "$RESULTS_ROOT" \
  "$HF_HOME" \
  "$(dirname "$EMBEDDING_CACHE")"

echo "Copying a fresh working tree..."
mkdir -p "$CODE_PATH"
rsync -a --delete \
  --exclude .git \
  --exclude .venv \
  --exclude .cache \
  --exclude results \
  "$SOURCE_PATH/" "$CODE_PATH/"

echo "Activating Python environment..."
eval "$(/storage/modules/packages/anaconda/bin/conda shell.bash hook)"
conda activate py313

echo "Installing experiment requirements..."
python -m pip install -q -r "$CODE_PATH/requirements-experiments.txt"

export HF_HOME
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

EXPERIMENT_MODULE="_experiments"
CONFIG_PATH="$CODE_PATH/$EXPERIMENT_MODULE/configs/oasst1_default.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Experiment config not found: $CONFIG_PATH"
  exit 3
fi
echo "Experiment module: $EXPERIMENT_MODULE"
echo "Experiment config: $CONFIG_PATH"

export OLLAMA_GPU_LAYERS=9999
export OLLAMA_FLASH_ATTENTION=1
echo "Starting Ollama..."
nohup ollama serve >"$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

for _ in {1..60}; do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null

ollama pull "$MODEL" | tee -a "$LOG_PATH/ollama_pull.log"
echo "Warm up GPU" | ollama run "$MODEL" >"$LOG_PATH/gpu-check.out" 2>>"$OLLAMA_LOG"

if ! grep -Eq 'offload library=CUDA|offloaded [1-9][0-9]*/[0-9]+ layers to GPU|CUDA0' "$OLLAMA_LOG"; then
  echo "Ollama did not report CUDA use. Inspect $OLLAMA_LOG."
  exit 2
fi
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" || true

cd "$CODE_PATH"
echo "Checking PyTorch CUDA access..."
srun python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

echo "Running all policy/capacity experiments..."
srun python -m "$EXPERIMENT_MODULE.run" \
  --config "$CONFIG_PATH" \
  --model "$MODEL" \
  --device cuda \
  --embedding-cache-path "$EMBEDDING_CACHE" \
  --output-dir "$RESULTS_ROOT" \
  --run-name "$RUN_NAME"

echo "Experiment completed successfully."
