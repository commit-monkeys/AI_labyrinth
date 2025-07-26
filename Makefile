.PHONY: install-dev

install-dev:
	python3 -m venv .venv
	. .venv/bin/activate && pip install uv
