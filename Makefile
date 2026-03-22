.PHONY: help install test run-all data baseline eval-raw finetune eval-finetuned llama report clean

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make test            - Run tests"
	@echo "  make run-all         - Run the entire pipeline"
	@echo "  make data            - Run data loading & preprocessing step"
	@echo "  make baseline        - Run the baseline model step"
	@echo "  make eval-raw        - Run raw transformer evaluation step"
	@echo "  make finetune        - Run transformer fine-tuning step"
	@echo "  make eval-finetuned  - Run fine-tuned transformer evaluation step"
	@echo "  make eval-ood        - Run Out-Of-Domain (ToxiGen) evaluation on fine-tuned models"
	@echo "  make llama           - Run LLaMA zero-shot evaluation step"
	@echo "  make report          - Generate final evaluation report"
	@echo "  make clean           - Remove cached files and outputs"

install:
	pip install -r requirements.txt

test:
	pytest tests/

run-all:
	python main.py --step all

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

report:
	python main.py --step report

clean:
	@python -c "import os, shutil, glob; ans=input('Are you sure you want to clear all trash (outputs, caches, etc.)? [y/n] '); exit(1) if ans.lower()!='y' else [shutil.rmtree(d, ignore_errors=True) for d in ['outputs','.pytest_cache','.hf_cache'] + glob.glob('**/__pycache__', recursive=True)]; print('Trash cleared successfully.')"
