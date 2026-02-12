.PHONY: docs

install:
	python -m pip install -e ".[dev]"
	pre-commit install

test:
	pytest -n auto --disable-warnings

precommit:
	pre-commit run

docs:
	mkdocs serve

docs-deploy:
	mkdocs gh-deploy

clean:
	rm -rf .pytest_cache build dist scikit_lego.egg-info .ipynb_checkpoints .coverage* .mypy_cache .ruff_cache

lint:
	ruff format sklego tests
	ruff check sklego tests --fix

build:
	uvx mobuild export nbs/ensemble sklego/ensemble

check: lint precommit test clean

pypi: clean
	uv build
	uv publish
