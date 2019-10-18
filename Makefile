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
	pytest --disable-warnings --cov=sklego
	rm -rf .coverage*
	pytest --nbval-lax doc/*.ipynb

check: flake test

docs:
	rm -rf doc/.ipynb_checkpoints
	sphinx-build doc docs

clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf docs
	rm -rf scikit_lego.egg-info
	rm -rf .ipynb_checkpoints
	rm -rf .coverage*

dist: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*