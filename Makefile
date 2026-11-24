.PHONY: up down logs build test test-docker backend-dev frontend-dev clean

up:            ## Build and start backend (:8000) and frontend (:3000)
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example - add your API keys to it")
	docker compose up --build -d
	@echo "Frontend: http://localhost:3000  Backend: http://localhost:8000/health"

down:          ## Stop the stack (the SQLite volume is kept)
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build

test:          ## Run backend tests locally
	cd backend && python -m pytest -q

test-docker:   ## Run backend tests inside the build image
	docker build --target test ./backend

backend-dev:
	cd backend && uvicorn app.main:create_app --factory --reload --port 8000

frontend-dev:
	cd frontend && npm install && npm run dev

clean:         ## Stop the stack and delete the SQLite volume
	docker compose down -v
