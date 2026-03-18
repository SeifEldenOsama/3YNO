.PHONY: help install run preview modal-run

help:
	@echo "install       Install dependencies"
	@echo "run           Run locally"
	@echo "preview       Preview outputs"
	@echo "modal-run     Run on Modal (H100)"

install:
	pip install -r requirements.txt

run:
	python scripts/run.py --lesson lesson.txt

preview:
	python scripts/preview.py

modal-run:
	modal run cloud/run.py
