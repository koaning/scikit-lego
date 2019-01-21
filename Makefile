flake:
	flake8 scikit-blocks
	flake8 tests

install:
	pip install -e .

develop:
	python setup.py develop

test:
	pytest
