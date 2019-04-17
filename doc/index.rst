.. scikit-lego documentation master file, created by
   sphinx-quickstart on Tue Mar 19 20:15:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scikit-lego
===========

.. image:: _static/logo.png

We love scikit learn but very often we find ourselves writing
custom transformers, metrics and models. The goal of this project
is to attempt to consolidate these into a package that offers
code quality/testing. This project is a collaboration between
multiple companies in the Netherlands. Note that we're not formally
affiliated with the scikit-learn project at all.

Installation
************

Install `scikit-lego` via pip with

.. code-block:: bash

   pip install scikit-lego


Alternatively you can fork/clone and run:

.. code-block:: bash

   pip install --editable .


Usage
*****

.. code-block:: python

   from sklego.transformers import RandomAdder

   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   from sklearn.pipeline import Pipeline

   ...

   mod = Pipeline([
       ("scale", StandardScaler()),
       ("random_noise", RandomAdder()),
       ("model", LogisticRegression(solver='lbfgs'))
   ])

   ...



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   contribution
   mixture-methods
   preprocessing
   api/modules
