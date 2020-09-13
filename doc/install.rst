Installation
============

.. warning:: This project is experimental and is in alpha. We
    do our best to keep things stable but you should assume that if
    you do not specify a version number that certain functionality
    can break.

Install **scikit-lego** via pip with

.. code-block:: bash

   pip install scikit-lego

Via conda with

.. code-block:: bash

   conda install -c conda-forge scikit-lego

Alternatively you can fork/clone and run:

.. code-block:: bash

   git clone https://github.com/koaning/scikit-lego
   pip install --editable .


Dependency installs
-------------------
Some functionality can only be used if certain dependencies are installed. This can be done by specifying the extra dependencies in square brackets after the package name.
Currently supported extras are **cvxpy** and **all** (which installs all extras). You can specify these as follows:

.. code-block:: bash

   pip install scikit-lego[cvxpy]

or from a local clone:

.. code-block:: bash

   pip install --editable ".[cvxpy]"
