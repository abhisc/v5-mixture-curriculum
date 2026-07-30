PY := .venv/bin/python
export PYTHONPATH := scripts:proxy

.PHONY: help check data agentic clean-data shards ablation tables all

help:
	@echo "make check      validate the spec and print the supply audit"
	@echo "make data       fetch corpora, synthesise agentic traces, clean, shard"
	@echo "make ablation   run the 13-run mixture ablation (~2h on an M-series GPU)"
	@echo "make tables     regenerate every table in README.md"

check:
	$(PY) scripts/budget.py
	$(PY) scripts/synthesis_cost.py

fetch:
	$(PY) proxy/fetch_data.py

agentic:
	$(PY) proxy/agentic_synth.py 2600
	cp proxy/data/raw/agentic/synthesis_stats.json results/synthesis_stats.json

clean-data:
	$(PY) proxy/clean.py

shards:
	$(PY) proxy/data_pipeline.py

data: fetch agentic clean-data shards

ablation:
	cd proxy && ../$(PY) -u run_ablation.py --steps 1200 --batch 32 --seq 384 --lr 6e-4 --seeds 2 \
		2>&1 | tee ../results/ablation.log

tables:
	$(PY) scripts/build_tables.py --inject

all: check data ablation tables
