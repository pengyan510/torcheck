test:
	poetry run pytest tests/

test-cov:
	poetry run pytest --cov=torcheck/ tests/

format:
	poetry run black torcheck/ tests/

lint:
	poetry run black --check torcheck/ tests/
	poetry run flake8 torcheck/ tests/

