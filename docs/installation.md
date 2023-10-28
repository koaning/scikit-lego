# Installation

!!! warning

    This project is experimental and is in alpha. We do our best to keep things stable but you should assume that if
    you do not specify a version number that certain functionality can break.

Install **scikit-lego**:

=== "pip"

    ```bash
    python -m pip install scikit-lego
    ```

=== "conda"

    ```bash
    conda install -c conda-forge scikit-lego
    ```

=== "source/git"

    ```bash
    python -m pip install git+https://github.com/koaning/scikit-lego.git
    ```

=== "local clone"

    ```bash
    git clone https://github.com/koaning/scikit-lego.git
    cd scikit-lego
    python -m pip install .
    ```

## Dependency installs

Some functionality can only be used if certain dependencies are installed. This can be done by specifying the extra dependencies in square brackets after the package name.

Currently supported extras are [**cvxpy**][cvxpy], [**formulaic**][formulaic], [**patsy**][patsy] and [**umap**][umap]. You can specify these as follows:

=== "pip"

    ```bash
    python -m pip install scikit-lego"[cvxpy]"
    python -m pip install scikit-lego"[formulaic]"
    python -m pip install scikit-lego"[patsy]"
    python -m pip install scikit-lego"[umap]"
    python -m pip install scikit-lego"[all]"
    ```

=== "local clone"

    ```bash
    git clone https://github.com/koaning/scikit-lego.git
    cd scikit-lego

    python -m pip install ".[cvxpy]"
    python -m pip install ."[formulaic]"
    python -m pip install ."[patsy]"
    python -m pip install ."[umap]"
    python -m pip install ".[all]"
    ```

[cvxpy]: https://www.cvxpy.org/
[formulaic]: https://matthewwardrop.github.io/formulaic/
[patsy]: https://patsy.readthedocs.io/en/latest/
[umap]: https://umap-learn.readthedocs.io/en/latest/index.html
