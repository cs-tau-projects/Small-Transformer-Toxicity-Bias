#!/bin/bash
#SBATCH --job-name=jigsaw_toxicity
#SBATCH --output=jigsaw_toxicity_%j.out
#SBATCH --error=jigsaw_toxicity_%j.err
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Exit on error
set -e

# Setup User specific output mapping
# You must provide YOUR username here, or we calculate it dynamically
USER_NAME=$(whoami)
BASE_OUTPUT_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/${USER_NAME}"

echo "Starting Toxicity Pipeline Job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Output Directory: $BASE_OUTPUT_DIR"
echo "Node: $SLURMD_NODENAME"

# Activate environment if we are using uv / venv. 
# Adjust this path if the virtual environment is stored elsewhere.
VENV_PATH="./.venv"
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment at $VENV_PATH..."
    source "$VENV_PATH/bin/activate"
else
    echo "Warning: No virtual environment found at $VENV_PATH. Using default system python."
fi

# We don't have access to this directory during dry-runs so we ensure it's created on the slurm node
mkdir -p "$BASE_OUTPUT_DIR"

# 1. Run Baseline
echo ""
echo "======================================"
echo "      RUNNING LOGISTIC BASELINE       "
echo "======================================"
PYTHONPATH=src python -m src.baseline --train

# 2. Run Training
echo ""
echo "======================================"
echo "          RUNNING DISTILBERT          "
echo "======================================"
# Pass the required base directory argument to train.py
python -m src.train --output_base_dir "$BASE_OUTPUT_DIR" --epochs 3 --batch_size 32

echo ""
echo "Job Completed Successfully!"
