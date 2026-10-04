# SentiNET developer tasks — `make help` lists them.
SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV   := backend/.venv
BIN    := $(abspath $(VENV)/bin)
PORT   ?= 8000

.PHONY: help setup setup-backend setup-frontend dev backend frontend test test-live lint fix typecheck \
        build serve analyze docker docker-up docker-down clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: setup-backend setup-frontend ## Install backend (venv) and frontend dependencies

setup-backend: ## Create backend/.venv and install Python deps
	@test -x $(BIN)/python || $(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r backend/requirements-dev.txt
	@test -f backend/.env || { cp backend/.env.example backend/.env; echo "created backend/.env (all settings optional)"; }

setup-frontend: ## Install frontend deps (npm ci)
	cd frontend && PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm ci --no-audit --no-fund

dev: ## API on :8000 (auto-reload) + Vite on :5173 (proxies /api) — Ctrl-C stops both
	@trap 'kill 0' INT TERM EXIT; \
	  (cd backend && $(BIN)/uvicorn app.main:app --reload --port $(PORT)) & \
	  (cd frontend && npm run dev) & \
	  wait

backend: ## API only, auto-reload
	cd backend && $(BIN)/uvicorn app.main:app --reload --port $(PORT)

frontend: ## Vite dev server only
	cd frontend && npm run dev

test: ## Offline backend test suite
	cd backend && $(BIN)/pytest -q

test-live: ## Smoke tests against real providers (network)
	cd backend && $(BIN)/pytest -q -m live

lint: ## Ruff (backend) + TypeScript typecheck (frontend)
	cd backend && $(BIN)/ruff check app tests
	cd frontend && npm run typecheck

fix: ## Apply ruff's safe autofixes
	cd backend && $(BIN)/ruff check --fix app tests

typecheck: ## Frontend typecheck only
	cd frontend && npm run typecheck

build: ## Production build of the web app (frontend/dist, served by the API)
	cd frontend && npm run build

serve: build ## Build the web app and serve everything on :8000
	cd backend && $(BIN)/uvicorn app.main:app --host 127.0.0.1 --port $(PORT)

analyze: ## Terminal intel: make analyze T=NVDA
	@test -n "$(T)" || { echo "usage: make analyze T=NVDA"; exit 2; }
	cd backend && $(BIN)/python -m app analyze $(T)

docker: ## Build the Docker image (sentinet:latest)
	docker build -t sentinet:latest .

docker-up: ## Run with docker compose (http://localhost:8000)
	docker compose up -d --build

docker-down: ## Stop the compose stack (data volume is kept)
	docker compose down

clean: ## Remove build output and caches (keeps venv, node_modules and data)
	rm -rf frontend/dist backend/.pytest_cache backend/.ruff_cache
	find backend -name __pycache__ -type d -prune -exec rm -rf {} +
