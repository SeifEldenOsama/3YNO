.PHONY: install local-inference modal-inference deploy-api

install:
	pip install -r requirements.txt

local-inference:
	python scripts/inference.py --image $(IMAGE) --prompt "$(PROMPT)"

modal-inference:
	modal run cloud/inference.py --image-path $(IMAGE) --prompt "$(PROMPT)"

deploy-api:
	modal deploy API/API.py
