#!/bin/bash
#SBATCH --job-name=toxicity-bias-4
#SBATCH --output=logs/toxicity_%j.out
#SBATCH --error=logs/toxicity_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48000

set -euo pipefail

# ── Storage Paths ───────────────────────────────────────
# All data, models, caches, and results go to the course storage directory
# (NOT the home dir — per assignment instructions)
COURSE_STORAGE="/vol/joberant_nobck/data/NLP_368307701_2526a/$(whoami)"
OUTPUT_DIR="${COURSE_STORAGE}/outputs"

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
# Temporarily disable 'nounset' because .bashrc often uses unset variables like PS1
set +u
source ~/.bashrc
conda activate venv
set -u
echo "✓ Conda env active  ($(python --version))"

# Try loading CUDA modules (Common on TAU CS Slurm)
echo "Attempting to load CUDA modules..."
module load cuda/12.4 || module load cuda/12.1 || module load cuda/11.8 || echo "⚠ No standard CUDA module found, relying on environment."

# Load HF token from .env (required for gated models like LLaMA)
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Loaded .env"
fi

# Point HuggingFace cache at course storage (avoids home quota issues)
export HF_HOME="${COURSE_STORAGE}/.hf_cache"
mkdir -p "$HF_HOME"
echo "✓ HF_HOME=$HF_HOME"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

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
echo "           STARTING: python main.py"
echo "═══════════════════════════════════════════════════"

python main.py --step all --output_dir "$OUTPUT_DIR"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ Pipeline completed at $(date)"
echo "═══════════════════════════════════════════════════"
