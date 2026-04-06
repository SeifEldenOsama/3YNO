.PHONY: help install generate modal-generate

help:
	@echo "install          Install dependencies"
	@echo "generate         Generate locally"
	@echo "modal-generate   Generate on Modal (H200)"

install:
	pip install -r requirements.txt

generate:
	python scripts/generate.py --frame frame.png --audio audio.wav --prompt "$(PROMPT)"

modal-generate:
	modal run cloud/generate.py --frame-path frame.png --audio-path audio.wav --prompt "$(PROMPT)"
