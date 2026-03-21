.PHONY: help install test run-all clean

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make test            - Run tests"
	@echo "  make run-all         - Run the entire pipeline"
	@echo "  make clean           - Remove cached files and outputs"

install:
	pip install -r requirements.txt

test:
	pytest tests/

run-all:
	python main.py --step all

clean:
	rm -rf outputs/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
