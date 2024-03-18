format:
	poetry run black .
	poetry run ruff check --fix --unsafe-fixes --fix-only .

lint:
	poetry run black --check .
	poetry run ruff check .
	poetry run pyright .
