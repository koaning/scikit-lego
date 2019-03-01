flake:
	flake8 skblocks
	flake8 tests

install:
	pip install -e .

develop:
	python setup.py develop

test:
	pytest

check: flake test