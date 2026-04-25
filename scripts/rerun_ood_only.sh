#!/bin/bash
#SBATCH --job-name=rerun-ood
#SBATCH --output=logs/rerun_ood_%j.out
#SBATCH --error=logs/rerun_ood_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48000

set -euo pipefail

# ── Storage Paths ───────────────────────────────────────
COURSE_STORAGE="/vol/joberant_nobck/data/NLP_368307701_2526a/$(whoami)"

TRANSFORMER_SEEDS=(42 44 46)
LLAMA_SEEDS=(42 44 46 67 89)

# ── Diagnostics ─────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Node      : $SLURMD_NODENAME"
echo "  Time      : $(date)"
echo "  Re-running OOD eval with unified subgroup names"
echo "═══════════════════════════════════════════════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Environment ─────────────────────────────────────────
set +u
source ~/.bashrc
conda activate venv
set -u
echo "✓ Conda env active  ($(python --version))"

module load cuda/12.4 || module load cuda/12.1 || module load cuda/11.8 || echo "⚠ No standard CUDA module found, relying on environment."

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Loaded .env"
fi

export HF_HOME="${COURSE_STORAGE}/.hf_cache"
mkdir -p "$HF_HOME" logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK

# ── Re-run transformer OOD eval per seed ────────────────
for SEED in "${TRANSFORMER_SEEDS[@]}"; do
    OUTPUT_DIR="${COURSE_STORAGE}/outputs/seed_${SEED}_run"
    echo ""
    echo "── Transformer OOD: seed=${SEED}  output=${OUTPUT_DIR} ──"
    make run-scientific ARGS="--step eval-ood --output_dir $OUTPUT_DIR --seed $SEED"
done

# ── Re-run LLaMA (data + eval-ood + llama) per seed ────
for SEED in "${LLAMA_SEEDS[@]}"; do
    OUTPUT_DIR="${COURSE_STORAGE}/outputs/llama_seed_${SEED}"
    echo ""
    echo "── LLaMA: seed=${SEED}  output=${OUTPUT_DIR} ──"
    make run-scientific ARGS="--step data  --output_dir $OUTPUT_DIR --seed $SEED"
    make run-scientific ARGS="--step eval-ood --output_dir $OUTPUT_DIR --seed $SEED"
    make run-scientific ARGS="--step llama --output_dir $OUTPUT_DIR --seed $SEED"
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ All OOD re-runs completed at $(date)"
echo "═══════════════════════════════════════════════════"
