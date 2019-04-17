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
	python -m doctest -v sklego/*.py

test: doctest
	pytest --disable-warnings

check: flake test

docs:
	sphinx-apidoc -f -o doc/api sklego
	sphinx-build doc docs

clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf scikit_lego.egg-info
	rm -rf .ipynb_checkpoints
