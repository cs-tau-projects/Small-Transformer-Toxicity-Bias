#!/bin/bash
#SBATCH --job-name=toxicity-sanity
#SBATCH --output=logs/sanity_%j.out
#SBATCH --error=logs/sanity_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16000

set -euo pipefail

# ── Argument Parsing ────────────────────────────────────
OUTPUT_SUBDIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --outputdir)
      OUTPUT_SUBDIR="$2"
      shift 2
      ;;
    *)
      if [ -z "$OUTPUT_SUBDIR" ] && [[ ! "$1" =~ ^- ]]; then
        OUTPUT_SUBDIR="$1"
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
echo "Conda env active  ($(python --version))"

echo "Attempting to load CUDA modules..."
module load cuda/12.4 || module load cuda/12.1 || module load cuda/11.8 || echo "No standard CUDA module found, relying on environment."

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

export HF_HOME="${COURSE_STORAGE}/.hf_cache"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

mkdir -p "$OUTPUT_DIR"

# ── CPU Pinning ─────────────────────────────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK

# ── GPU Diagnostics ─────────────────────────────────────
echo ""
echo "── GPU Diagnostics ──"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-'NOT SET'}"
nvidia-smi --query-gpu=name,index,memory.total --format=csv,noheader
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
echo "─────────────────────"
echo ""

# ── Ensure data splits exist ───────────────────────────
# The sanity check needs the pre-saved data splits. Run the
# data step first (skips automatically if splits already exist).
echo "Ensuring data splits exist..."
python main.py --step data --output_dir "$OUTPUT_DIR"

# ── Run Sanity Check ───────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "   STARTING: Sanity Check / Overfitting Experiment"
echo "═══════════════════════════════════════════════════"

python -m src.sanity_check \
    --output_dir "$OUTPUT_DIR" \
    --n_samples 100 \
    --epochs 20 \
    --seed 42

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Sanity check completed at $(date)"
echo "═══════════════════════════════════════════════════"
