.PHONY: install pipeline test lint gates api dashboard advanced docker
install:   ; pip install -r requirements-dev.txt
pipeline:  ; python -m src.pipeline
test:      ; python -m pytest -q
lint:      ; ruff check .
gates:     ; python -m src.benchmark && python -m src.gates
api:       ; uvicorn backend.api:app --reload
dashboard: ; streamlit run dashboard/dashboard.py
# long-running evidence jobs (about 30 min on one core): real data, bootstrap CIs, multi-seed, drift backtest, tuning
advanced:  ; python -m src.realdata --download && python -m src.robustness && python -m src.backtest && python -m src.tune
docker:    ; docker compose up --build
