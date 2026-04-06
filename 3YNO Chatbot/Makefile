.PHONY: install chat modal-chat deploy

install:
	pip install -r requirements.txt

chat:
	python scripts/chat.py

modal-chat:
	modal run cloud/run.py --message "$(MSG)"

deploy:
	modal deploy API/API.py
