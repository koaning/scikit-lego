.PHONY: docs

flake:
	flake8 sklego
	flake8 tests

install:
	pip install -e .

develop:
	python setup.py develop

test:
	pytest

check: flake test

docs:
	sphinx-apidoc -f -o doc/api sklego
	sphinx-build doc docs