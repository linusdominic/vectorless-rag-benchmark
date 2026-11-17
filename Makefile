.PHONY: install test lint fetch summarize questions bench report spend

install:
	uv sync

test: ## no network or API key needed
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fetch: ## download Regulation Z from eCFR and parse it into data/regz.json
	uv run vrbench fetch

summarize: ## paid: write outline summaries with the navigator model
	uv run vrbench summarize

questions: ## paid: generate the 60-question evaluation set
	uv run vrbench questions --count 60 --seed 7

bench: ## paid: run all six methods over the question set
	uv run vrbench run

report: ## free: rebuild results/report.md from results/records.jsonl
	uv run vrbench report --write

spend: ## show what has been spent so far
	uv run vrbench spend
