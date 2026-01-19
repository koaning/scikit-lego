# Docs readme

The docs folder contains the documentation for the scikit-lego package.

The documentation is generated using [Material for MkDocs][mkdocs-material], its extensions and a few plugins.
In particular the `mkdocstrings-python` is used for API rendering.

## Render locally

To render the documentation locally, you can run the following command from the root of the repository:

```console
make docs
```

Then the documentation page will be available at [localhost][localhost].

## Remark

The majority of code and code generate plots in the documentation is generated using the scripts in the `docs/_scripts` folder,
and accessed via the [pymdown snippets][pymdown-snippets] extension.

To generate the plots from scratch it is enough to run the following command from the root of the repository:

```console
cd docs
make generate-all
```

which will run all the scripts and save results in the `docs/_static` folder.

[mkdocs-material]: https://squidfunk.github.io/mkdocs-material/
[pymdown-snippets]: https://facelessuser.github.io/pymdown-extensions/extensions/snippets/
[localhost]: http://localhost:8000/

## NLP Example

An example demonstrating text classification using TF-IDF and Logistic Regression is available in:

docs/examples/nlp_text_classification.py
