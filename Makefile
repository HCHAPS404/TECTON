.PHONY: doctor test smoke baseline catboost export lock dashboard-build dashboard-test dashboard-open
doctor:
	uv run --frozen python -m tecton doctor
test:
	uv run --frozen python -m unittest discover -s tests -v
	uv run --frozen ruff check src tests scripts
smoke:
	uv run --frozen python -m tecton synthetic
	uv run --frozen python -m tecton run --config configs/smoke.json
baseline:
	uv run --frozen python -m tecton run --config configs/baseline.json
catboost:
	uv run --frozen python -m tecton run --config configs/catboost.json
export:
	uv run --frozen python -m tecton bundle
lock:
	uv lock
	uv export --frozen --no-dev --no-emit-project --no-hashes --format requirements.txt --output-file requirements-colab.txt
dashboard-build:
	cd dashboard && npm ci && npm run build
dashboard-test:
	cd dashboard && npm test
dashboard-open:
	python3 -m http.server 8765 --bind 127.0.0.1 --directory dashboard/dist
