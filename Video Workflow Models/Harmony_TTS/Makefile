.PHONY: help install train inference upload modal-train modal-inference

help:
	@echo "install          Install dependencies"
	@echo "train            Train locally"
	@echo "inference        Run inference locally"
	@echo "upload           Upload model to HuggingFace"
	@echo "modal-train      Train on Modal (H100)"
	@echo "modal-inference  Run inference on Modal"

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

inference:
	python scripts/inference.py --text "$(TEXT)" --description "$(DESC)"

upload:
	python scripts/upload.py --path ./outputs/model

modal-train:
	modal run cloud/train.py

modal-inference:
	modal run cloud/inference.py --text "$(TEXT)"
