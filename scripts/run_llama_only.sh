#!/bin/bash
#SBATCH --job-name=llama-full-eval
#SBATCH --output=logs/llama_full_%j.out
#SBATCH --error=logs/llama_full_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48000

set -euo pipefail

# --- Argument Parsing ---
OUTPUT_SUBDIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --outputdir) OUTPUT_SUBDIR="$2"; shift 2 ;;
    *) if [ -z "$OUTPUT_SUBDIR" ]; then OUTPUT_SUBDIR="$1"; fi; shift ;;
  esac
done

# --- Storage Paths ---
COURSE_STORAGE="/vol/joberant_nobck/data/NLP_368307701_2526a/$(whoami)"
if [ -n "$OUTPUT_SUBDIR" ]; then
    OUTPUT_DIR="${COURSE_STORAGE}/outputs/${OUTPUT_SUBDIR}"
else
    OUTPUT_DIR="${COURSE_STORAGE}/outputs"
fi
HF_HOME="${COURSE_STORAGE}/.hf_cache"

# --- Environment Setup ---
set +u
source ~/.bashrc
conda activate venv
set -u

module load cuda/12.4 || module load cuda/12.1 || module load cuda/11.8 || echo "Warning: No CUDA module found"

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

export HF_HOME="$HF_HOME"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

# --- Execution ---
echo "Starting Data and LLaMA steps on full dataset..."

# We use make run-scientific because it sets --eval_samples -1 automatically.
# We run them sequentially to ensure data is ready before LLaMA starts.

echo "Running Data Preparation..."
make run-scientific ARGS="--step data --output_dir $OUTPUT_DIR $@"

echo "Running LLaMA Evaluation..."
make run-scientific ARGS="--step llama --output_dir $OUTPUT_DIR $@"

echo "Completed at $(date)"
