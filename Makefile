.PHONY: help install test run-all tiny-run data baseline eval-raw finetune eval-finetuned eval-ood llama analysis report clean lint format check slurm

PYTHON = python

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make test            - Run tests"
	@echo "  make lint            - Check code quality with ruff"
	@echo "  make format          - Format code with ruff"
	@echo "  make run-all         - Run the entire pipeline"
	@echo "  make tiny-run        - Quick CPU-friendly validation run (100 samples)"
	@echo "  make data            - Run data loading & preprocessing step"
	@echo "  make baseline        - Run the baseline model step"
	@echo "  make eval-raw        - Run raw transformer evaluation step"
	@echo "  make finetune        - Run transformer fine-tuning step"
	@echo "  make eval-finetuned  - Run fine-tuned transformer evaluation step"
	@echo "  make eval-ood        - Run Out-Of-Domain (ToxiGen) evaluation"
	@echo "  make llama           - Run LLaMA zero-shot evaluation step"
	@echo "  make analysis        - Run dataset statistics and error sampling"
	@echo "  make hf-login        - Authenticate with Hugging Face Hub"
	@echo "  make report          - Generate final evaluation report"
	@echo "  make slurm           - Submit the pipeline as a Slurm job (sbatch)"
	@echo "  make clean           - Remove cached files and outputs"

install:
	pip install -r requirements.txt

test:
	pytest tests/

lint:
	ruff check .

format:
	ruff format .

check: lint test

small-run:
	python main.py --models distilbert-base-uncased --train_samples 20000 --eval_samples 5000

tiny-run:
	$(MAKE) run-all ARGS="--train_samples 100 --eval_samples 100 $(ARGS)"

run-all:
	python main.py --step all $(ARGS)

run-scientific:
	@echo "Running with Scientific Sampling (20k Train, Full Test)..."
	$(MAKE) run-all ARGS="--train_samples 20000 --eval_samples -1 $(ARGS)"

run-full:
	@echo "Running on the ENTIRE 1.8M dataset (Warning: This will take several days)..."
	$(MAKE) run-all ARGS="--train_samples -1 --eval_samples -1 $(ARGS)"

data:
	python main.py --step data

baseline:
	python main.py --step baseline

eval-raw:
	python main.py --step eval-raw

finetune:
	python main.py --step finetune

eval-finetuned:
	python main.py --step eval-finetuned

eval-ood:
	python main.py --step eval-ood

llama:
	python main.py --step llama

analysis:
	python main.py --step analysis

hf-login:
	hf auth login

report:
	python main.py --step report

slurm:
	@mkdir -p logs
	sbatch scripts/run_all_job.sh
	@echo "Monitor with: squeue --me"

slurm-tiny:
	@mkdir -p logs
	@echo "Submitting quick validation job..."
	sbatch scripts/tiny_run_job.sh

conda-fix-torch:
	@echo "Ensuring PyTorch is installed with CUDA support in the active conda environment..."
	conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

clean:
	@python -c "import os, shutil, glob; ans=input('Are you sure you want to clear all trash (outputs, caches, etc.)? [y/n] '); exit(1) if ans.lower()!='y' else [shutil.rmtree(d, ignore_errors=True) for d in ['outputs','.pytest_cache','.hf_cache','.ruff_cache'] + glob.glob('**/__pycache__', recursive=True)]; print('Trash cleared successfully.')"
