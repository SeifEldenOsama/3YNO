# ─────────────────────────────────────────────────────────────
#  FLUX LoRA — Makefile
#  Usage:  make <target>
# ─────────────────────────────────────────────────────────────

.PHONY: help install train inference upload download modal-train modal-inference

# ── Default ───────────────────────────────────────────────────
help:
	@echo ""
	@echo "  FLUX LoRA Fine-tuning"
	@echo "  ───────────────────────────────────────────────"
	@echo "  make install          Install dependencies"
	@echo "  make train            Train locally"
	@echo "  make inference        Run inference locally"
	@echo "  make upload           Upload LoRA to HuggingFace"
	@echo "  make download         Download LoRA from HuggingFace"
	@echo "  make modal-train      Train on Modal cloud (H100)"
	@echo "  make modal-inference  Run inference on Modal cloud"
	@echo "  ───────────────────────────────────────────────"
	@echo ""

# ── Install ───────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Local runs ────────────────────────────────────────────────
train:
	python scripts/train.py --config config.yaml

inference:
	python scripts/inference.py --config config.yaml

upload:
	python scripts/upload.py --config config.yaml

download:
	python scripts/upload.py --config config.yaml --download

# ── Modal runs ────────────────────────────────────────────────
modal-train:
	modal run cloud/train.py

modal-inference:
	modal run cloud/inference.py

modal-inference-custom:
	modal run cloud/inference.py \
		--prompt "$(PROMPT)" \
		--num-images $(or $(N),4) \
		--seed $(or $(SEED),42)
