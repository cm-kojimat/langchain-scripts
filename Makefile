format:
	poetry run black .
	poetry run ruff check --fix --unsafe-fixes --fix-only .

lint:
	poetry run black --check .
	poetry run ruff check .
	poetry run pyright .

test:
	poetry run pytest --cov=. --cov-report=html --cov-report=term -vv .
