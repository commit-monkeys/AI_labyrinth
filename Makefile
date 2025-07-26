.PHONY: install-dev

install-dev:
	python3 -m venv .venv
	. .venv/bin/activate && pip install uv && uv pip install pre-commit
	. .venv/bin/activate && pre-commit install
