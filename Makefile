SHELL := /bin/bash

VENV = ising_model
PYTHON = $(VENV)/bin/python
MAIN = main.py

banner:
	@echo "  ██████████   █████████  ███████████ ██████   ██████"
	@echo " ░░███░░░░░█  ███░░░░░███░░███░░░░░░█░░██████ ██████ "
	@echo "  ░███  █ ░  ███     ░░░  ░███   █ ░  ░███░█████░███ "
	@echo "  ░██████   ░███          ░███████    ░███░░███ ░███ "
	@echo "  ░███░░█   ░███          ░███░░░█    ░███ ░░░  ░███ "
	@echo "  ░███ ░   █░░███     ███ ░███  ░     ░███      ░███ "
	@echo " ██████████ ░░█████████  █████       █████     ██████"
	@echo "░░░░░░░░░░   ░░░░░░░░░  ░░░░░       ░░░░░     ░░░░░  "

create:
	python3 -m venv ising_model

run:
	@echo "Running simulation..."
	@source $(VENV)/bin/activate && \
	$(PYTHON) $(MAIN)

install:
	ising_model/bin/pip install -r requirements.txt

build: banner create install run 