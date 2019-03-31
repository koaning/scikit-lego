.PHONY: docs

flake:
	flake8 sklego
	flake8 tests
	flake8 setup.py

install:
	pip install -e .

develop:
	pip install -e ".[dev]"
	python setup.py develop


doctest:
	python -m doctest sklego/*.py

test: doctest
	pytest

check: flake test

docs:
	sphinx-apidoc -f -o doc/api sklego
	sphinx-build doc docs
	touch docs/.nojekyll

clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf scikit_lego.egg-info
