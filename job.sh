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
# Activate conda environment (required — slurm jobs start in a fresh shell)
source ~/.bashrc
conda activate venv
echo "✓ Conda env active  ($(python --version))"

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

# ── Run Pipeline ────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "           STARTING: make run-all"
echo "═══════════════════════════════════════════════════"

python main.py --step all --output_dir "$OUTPUT_DIR"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ Pipeline completed at $(date)"
echo "═══════════════════════════════════════════════════"
