.PHONY: install models demo backend web test

install:            ## Python + Node dependencies
	pip install -e "ml[dev]" -e "backend[dev]"
	cd web && npm install

models:             ## MediaPipe models for extraction (Python) and live tracking (browser)
	./ml/scripts/download_mediapipe_models.sh

demo:               ## synthetic model + sign library so the app runs before real data
	./scripts/demo.sh

backend:            ## API on :8000 (set ANTHROPIC_API_KEY for Claude; otherwise a rule-based fallback)
	cd backend && uvicorn app.main:app --reload --port 8000

web:                ## web app on :3000
	cd web && npm run dev

test:               ## all tests (the ML end-to-end test trains a tiny model, about a minute)
	cd ml && python -m pytest -q
	cd backend && python -m pytest -q
	cd web && npm test
