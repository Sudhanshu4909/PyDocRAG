.PHONY: help install setup test run docker-up docker-down clean

help:
	@echo "PyDocRAG - Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Set up infrastructure (Docker)"
	@echo "  make migrate     - Migrate docs to Qdrant"
	@echo "  make test        - Run tests"
	@echo "  make run         - Run API server"
	@echo "  make docker-up   - Start all services with Docker"
	@echo "  make docker-down - Stop all services"
	@echo "  make clean       - Clean up generated files"

install:
	pip install -r requirements.txt

setup:
	docker run -d -p 6333:6333 -v $(PWD)/qdrant_storage:/qdrant/storage qdrant/qdrant
	docker run -d -p 6379:6379 redis:alpine
	cp .env.example .env
	@echo "✓ Infrastructure started. Edit .env file as needed."

migrate:
	python scripts/migrate_to_qdrant.py

test:
	pytest tests/ -v

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
