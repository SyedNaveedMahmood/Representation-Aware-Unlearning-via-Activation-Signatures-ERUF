# ERUF Pipeline Runner

This repo runs the full pipeline sequentially from a single script, including the self-healing loop (Module 7 → Module 8) until forgetting thresholds hold.

## Install
```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

## Run the full pipeline
```bash
python run_pipeline.py
```

Outputs are written under `outputs/`.

Optional loop controls (env vars):
```bash
FORGET_SMR_THRESHOLD=0.05 \
FORGET_EL10_THRESHOLD=1.0 \
FORGET_CONFIRM_ROUNDS=10 \
FORGET_MAX_DAYS=3 \
python run_pipeline.py
```
