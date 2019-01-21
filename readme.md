# scikit-blocks

We love scikit learn but very often we find ourselves writing
custom transformers, metrics and models. The goal of this project
is to attempt to consolidate these into a package that offers 
code quality/testing. This project is a collaboration between
multiple companies in the Netherlands. 

## project structure 

```
│
├── notebooks/          <- Jupyter notebooks. Naming convention is a short `-` delimited 
│                          description, a number (for ordering), and the creator's initials,
│                          e.g. `initial-data-exploration-01-hg`.
├── tests/              <- Unit tests.
├── skblocks/           <- Python module with source code of this project.
├── Makefile            <- Makefile with commands like `make environment`
└── README.md           <- The top-level README for developers using this project.
```

## installation 

Install `scikit-blocks` in the virtual environment via:

```bash
$ pip install --editable .
```

## usage 

```python
from skblocks.transformers import RandomAdder
```
