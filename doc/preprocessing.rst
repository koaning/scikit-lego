Preprocessing
=============

There are many preprocessors in scikit-lego and in this document we
would like to highlight a few such that you might be inspired to use
pipelines a little bit more flexibly.

Column Capping
**************

Some models are great at interpolation but less good at extrapolation.
One way to potentially circumvent this problem is by capping extreme
valeus that occur in the dataset **X**.

.. image:: _static/column-capper.png

To see how they work we demonstrate a few examples below.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from sklego.preprocessing import ColumnCapper

    np.random.seed(42)
    X = np.random.uniform(0, 1, (100000, 2))

    cc = ColumnCapper()
    output = cc.fit(X).transform(X)
    output.min(axis=0) # array([0.05120598, 0.0502972 ])
    output.max(axis=0) # array([0.95030677, 0.95088171])

    cc = ColumnCapper(quantile_range=(10, 90))
    output = cc.fit(X).transform(X)
    output.min(axis=0) # array([0.10029693, 0.09934085])
    output.max(axis=0) # array([0.90020412, 0.89859006])

Note that the column capper does not deal with missing values
but it does support pandas dataframes as well as infinite values.

.. code-block:: python

    arr = np.array([[0.0, np.inf], [-np.inf, 1.0]])
    cc.transform(arr) # array([[0.10029693, 0.89859006], [0.10029693, 0.89859006]])


Patsy Formulas
**************

If you're used to the statistical programming language R you might have
seen a formula object before. This is an object that represents a shorthand
way to design variables used in a statistical model. The python project patsy_
took this idea and made it available for python. From sklego we've made a
wrapper such that you can also use these in your pipelines.

.. code-block:: python

    import pandas as pd
    from sklego.preprocessing import PatsyTransformer

    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": ["yes", "yes", "no", "maybe", "yes"],
                       "y": [2, 2, 4, 4, 6]})
    X, y = df[["a", "b"]], df[["y"]].values
    pt = PatsyTransformer("a + np.log(a) + b")
    pt.fit(X, y).transform(X)

This will result in the following array:

.. code-block:: python

    array([[1.        , 0.        , 1.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 2.        , 0.69314718],
           [1.        , 1.        , 0.        , 3.        , 1.09861229],
           [1.        , 0.        , 0.        , 4.        , 1.38629436],
           [1.        , 0.        , 1.        , 5.        , 1.60943791]])


You might notice that the first column contains the constant array
equal to one. You might also expect 3 dummy variable columns instead of 2.
This is because the design matrix from patsy attempts to keep the
columns in the matrix linearly independant of eachother.

If this is not something you'd want to create you can choose to omit
it by indicating "-1" in the formula.

.. code-block:: python

    pt = PatsyTransformer("a + np.log(a) + b - 1")
    pt.fit(X, y).transform(X)

This will result in the following array:

.. code-block:: python

    array([[0.        , 0.        , 1.        , 1.        , 0.        ],
           [0.        , 0.        , 1.        , 2.        , 0.69314718],
           [0.        , 1.        , 0.        , 3.        , 1.09861229],
           [1.        , 0.        , 0.        , 4.        , 1.38629436],
           [0.        , 0.        , 1.        , 5.        , 1.60943791]])

You'll notice that now the constant array is gone and it is replaced with
a dummy array. Again this is now possible because patsy wants to guarantee
that each column in this matrix is linearly independant of eachother.

The formula syntax is pretty powerful, if you'd like to learn we refer you
to formulas_ documentation.

.. _patsy: https://patsy.readthedocs.io/en/latest/
.. _formulas: https://patsy.readthedocs.io/en/latest/formulas.html
