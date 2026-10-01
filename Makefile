# kyrgyz-tts-mini — run `make help` for all targets

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

PYTHON  ?= $(shell for p in python3.12 python3.13 python3.11 python3; do command -v $$p >/dev/null 2>&1 && { echo $$p; break; }; done)
VENV    := .venv
PY      := $(VENV)/bin/python
CLI     := $(PY) -m kyrgyz_tts_mini
STAMP   := $(VENV)/.installed

TEXT    ?= Саламатсызбы! Бүгүн аба ырайы абдан жакшы.
VOICE   ?= woman
FILE    ?= samples/texts.txt
ARGS    ?=

SPEAK   = $(CLI) speak --voice $(VOICE) $(ARGS)

RUFF        := $(VENV)/bin/ruff
RUFF_OPTS   := --line-length 120 --extend-exclude kyrgyz_tts_mini/acoustic,kyrgyz_tts_mini/vocoder
RUFF_CHECK  := $(RUFF) check $(RUFF_OPTS) --select E,F,I,B,UP --ignore E501 --target-version py311
RUFF_FORMAT := $(RUFF) format $(RUFF_OPTS)

.PHONY: help setup install download check doctor run speak say speak-file web demo upload \
        test test-fast lint format clean clean-outputs clean-all clean-models

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- setup -----------------------------------------------------------------------

setup: install download doctor ## One-time setup: environment + models + health check

$(PY):
	@test -n "$(PYTHON)" || { echo "error: Python 3.11+ not found; install it or run: make setup PYTHON=/path/to/python3.12" >&2; exit 1; }
	@$(PYTHON) -c "import sys, venv, ensurepip; sys.exit(sys.version_info < (3, 11))" || { echo "error: need Python 3.11+ with venv (Debian/Ubuntu: sudo apt install python3-venv)" >&2; exit 1; }
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip

$(STAMP): $(PY) requirements.txt
	$(PY) -m pip install --quiet -r requirements.txt
	@touch $@

install: $(STAMP) ## Create .venv and install (re-runs when requirements.txt changes)

download: $(STAMP) ## Download missing models from Hugging Face (ARGS=--force to re-download)
	$(CLI) download $(ARGS)

check: $(STAMP) ## Verify the installed models against their checksums
	$(CLI) download --check

upload: $(STAMP) ## Upload local models to Hugging Face (run `.venv/bin/hf auth login` first; ARGS=--public)
	$(CLI) upload $(ARGS)

doctor: $(STAMP) ## Check environment + models and run a test synthesis
	$(CLI) doctor

# --- play ------------------------------------------------------------------------

run: $(STAMP) ## Say the same sentence in both voices → outputs/ (PLAY=1 to listen)
	for v in woman man; do $(CLI) speak "$(TEXT)" --voice $$v -o outputs/demo-$$v.wav $(if $(PLAY),--play); done

speak: $(STAMP) ## Speak text: make speak TEXT="Салам" VOICE=man ARGS=--play
	$(SPEAK) "$(TEXT)"

say: $(STAMP) ## Interactive: type a line, hear it
	$(SPEAK)

speak-file: $(STAMP) ## Speak every line of a text file: make speak-file FILE=story.txt
	$(SPEAK) --file "$(FILE)"

web: $(STAMP) ## Web UI at http://127.0.0.1:7860 (ARGS=--share for a public link)
	$(CLI) web $(ARGS)

demo: override ARGS += --open
demo: web ## Gradio playground, opened in the browser (ARGS=--share for a public link)

# --- develop ---------------------------------------------------------------------

test: $(STAMP) ## Run all tests (model tests skip when models are missing)
	$(PY) -m pytest $(ARGS)

test-fast: $(STAMP) ## Run the tests that need no models
	$(PY) -m pytest --ignore tests/test_engine.py $(ARGS)

lint: $(STAMP) ## Lint and check formatting
	$(RUFF_CHECK) .
	$(RUFF_FORMAT) --check .

format: $(STAMP) ## Fix lint issues and format
	$(RUFF_CHECK) --fix .
	$(RUFF_FORMAT) .

# --- clean -----------------------------------------------------------------------

clean: ## Remove caches
	find . -path ./$(VENV) -prune -o \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -type d -print -exec rm -rf {} +

clean-outputs: ## Remove generated audio in outputs/
	rm -rf outputs

clean-all: clean clean-outputs ## Also remove .venv (models/ are kept)
	rm -rf $(VENV)

clean-models: ## Delete downloaded models (asks for CONFIRM=yes)
	@[ "$(CONFIRM)" = yes ] || { echo "This deletes models/ ($$(du -sh models 2>/dev/null | cut -f1 || echo 0)). To proceed: make clean-models CONFIRM=yes"; exit 1; }
	rm -rf models
