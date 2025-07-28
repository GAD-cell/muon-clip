PYTHON_VERSION := 3.11
VENV  := .venv


TEST_SCRIPT := src/test/test.py

.PHONY: env test

env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up eval environment..."
	@uv venv $(VENV) --python $(PYTHON_VERSION) --no-project
	@uv pip install -r requirements.txt --python $(VENV)/bin/python
	@echo "Evaluation environment ready."


test:
	@python3 $(TEST_SCRIPT)