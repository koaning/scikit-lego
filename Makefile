.PHONY: docs

flake:
	flake8 sklego
	flake8 tests
	flake8 setup.py

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"
	pre-commit install

doctest:
	python -m doctest -v sklego/*.py

test-notebooks:
	pytest --nbval-lax doc/*.ipynb

test: doctest
	pytest --disable-warnings --cov=sklego
	rm -rf .coverage*
	pytest --nbval-lax doc/*.ipynb

precommit:
	pre-commit run

spelling:
	codespell sklego/*.py

docs:
	rm -rf doc/.ipynb_checkpoints
	sphinx-build -a -E doc docs

docs-deploy: docs
	netlify deploy --dir=docs --prod

clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf docs
	rm -rf scikit_lego.egg-info
	rm -rf .ipynb_checkpoints
	rm -rf .coverage*

black:
	black sklego tests setup.py

check: flake precommit test spelling clean

pypi: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*
