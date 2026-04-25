#!/bin/bash
#SBATCH --job-name=llama-data-full
#SBATCH --output=logs/llama_%j.out
#SBATCH --error=logs/llama_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48000

set -euo pipefail

# ── Argument Parsing ────────────────────────────────────
OUTPUT_SUBDIR=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --outputdir)
      OUTPUT_SUBDIR="$2"
      shift 2
      ;;
    *)
      if [ -z "$OUTPUT_SUBDIR" ] && [[ ! "$1" =~ ^- ]]; then
        OUTPUT_SUBDIR="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

# ── Storage Paths ───────────────────────────────────────
COURSE_STORAGE="/vol/joberant_nobck/data/NLP_368307701_2526a/$(whoami)"

if [ -n "$OUTPUT_SUBDIR" ]; then
    OUTPUT_DIR="${COURSE_STORAGE}/outputs/${OUTPUT_SUBDIR}"
else
    OUTPUT_DIR="${COURSE_STORAGE}/outputs"
fi

# ── Diagnostics ─────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Node      : $SLURMD_NODENAME"
echo "  Partition : $SLURM_JOB_PARTITION"
echo "  GPUs      : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Time      : $(date)"
echo "  Working   : $(pwd)"
echo "  Storage   : $COURSE_STORAGE"
echo "  Output    : $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Environment ─────────────────────────────────────────
set +u
source ~/.bashrc
conda activate venv
set -u
echo "✓ Conda env active  ($(python --version))"

echo "Attempting to load CUDA modules..."
module load cuda/12.4 || module load cuda/12.1 || module load cuda/11.8 || echo "⚠ No standard CUDA module found, relying on environment."

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Loaded .env"
fi

export HF_HOME="${COURSE_STORAGE}/.hf_cache"
mkdir -p "$HF_HOME"
echo "✓ HF_HOME=$HF_HOME"

mkdir -p "$OUTPUT_DIR"

# ── CPU Pinning (prevent sklearn/joblib from over-subscribing) ──
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK

# ── Diagnostics (GPU/Torch) ─────────────────────────────
echo ""
echo "── GPU Diagnostics ──"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-'NOT SET'}"
nvidia-smi --query-gpu=name,index,memory.total --format=csv,noheader
python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'CUDA Version: {torch.version.cuda}')"
echo "─────────────────────"
echo ""

# ── Run Pipeline ────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "           STARTING: Data + LLaMA (Full Run)"
echo "═══════════════════════════════════════════════════"

# Step 1: Prepare Full Data
make run-scientific ARGS="--step data --output_dir $OUTPUT_DIR ${EXTRA_ARGS[*]:-}"

# Step 2: Prepare OOD Data
# This generates the standardized ToxiGen parquet file for LLaMA evaluation
make run-scientific ARGS="--step eval-ood --output_dir $OUTPUT_DIR ${EXTRA_ARGS[*]:-}"

# Step 3: Run LLaMA Evaluation
make run-scientific ARGS="--step llama --output_dir $OUTPUT_DIR ${EXTRA_ARGS[*]:-}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ LLaMA Pipeline completed at $(date)"
echo "═══════════════════════════════════════════════════"
