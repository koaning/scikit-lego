.PHONY: docs

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest --disable-warnings --cov=sklego
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
	ruff check sklego tests setup.py --fix
	ruff format sklego tests setup.py

check: precommit test clean

pypi: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*
