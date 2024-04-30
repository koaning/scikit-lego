.PHONY: docs

install:
	python -m pip install -e ".[dev]"
	pre-commit install

test:
	pytest -n auto --disable-warnings --cov=sklego
	rm -rf .coverage*

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

check: lint precommit test clean

pypi: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*
