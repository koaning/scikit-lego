Meta Models
===========

Certain models in scikit-lego are "meta". Meta models are
models that depend on other estimators that go in and these
models will add features to the input model. One way of thinking
of a meta model is to consider it to be a way to "decorate" a
model.

This part of the documentation will highlight a few of them.

Grouped Estimation
------------------

A kind introduction to "meta"-models is the `GroupedEstimator`. To
help explain what it can do we'll consider three models to predict
the chicken dataset. The chicken data has 578 rows and 4 columns
from an experiment on the effect of diet on early growth of chicks.
The body weights of the chicks were measured at birth and every second
day thereafter until day 20. They were also measured on day 21.
There were four groups on chicks on different protein diets.

Setup
*****

Let's first load a bunch of things to do this.

.. code-block::python

    import numpy as np
    import pandas as pd
    import matplotlib.pylab as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    from sklego.datasets import load_chicken
    from sklego.preprocessing import ColumnSelector

    df = load_chicken(give_pandas=True)

    def plot_model(model):
        df = load_chicken(give_pandas=True)
        model.fit(df[['diet', 'time']], df['weight'])
        metric_df = df[['diet', 'time', 'weight']].assign(pred=lambda d: model.predict(d[['diet', 'time']]))
        metric = mean_absolute_error(metric_df['weight'], metric_df['pred'])
        plt.scatter(df['time'], df['weight'])
        for i in [1, 2, 3, 4]:
            pltr = metric_df[['time', 'diet', 'pred']].drop_duplicates().loc[lambda d: d['diet'] == i]
            plt.plot(pltr['time'], pltr['pred'], color='.rbgy'[i])
        plt.title(f"linear model per group, MAE: {np.round(metric, 2)}");

Model 1: Linear Regression with Dummies
***************************************

.. code-block:: python

    feature_pipeline = Pipeline([
        ("datagrab", FeatureUnion([
             ("discrete", Pipeline([
                 ("grab", ColumnSelector("diet")),
                 ("encode", OneHotEncoder(categories="auto", sparse=False))
             ])),
             ("continous", Pipeline([
                 ("grab", ColumnSelector("time")),
                 ("standardize", StandardScaler())
             ]))
        ]))
    ])

    pipe = Pipeline([
        ("transform", feature_pipeline),
        ("model", LinearRegression())
    ])

    plot_model(pipe)

This code results in a model that is demonstrated below.

Model 2: Linear Regression in GroupedEstimation
***********************************************

.. code-block:: python

    from sklego.meta import GroupedEstimator
    mod = GroupedEstimator(LinearRegression(), groups=["diet"])
    plot_model(mod)

This code results in a model that is demonstrated below.

Model 3: Dummy Regression in GroupedEstimation
**********************************************

.. code-block:: python

    from sklearn.dummy import DummyRegressor

    feature_pipeline = Pipeline([
        ("datagrab", FeatureUnion([
             ("discrete", Pipeline([
                 ("grab", ColumnSelector("diet")),
             ])),
             ("continous", Pipeline([
                 ("grab", ColumnSelector("time")),
             ]))
        ]))
    ])

    pipe = Pipeline([
        ("transform", feature_pipeline),
        ("model", GroupedEstimator(DummyRegressor(strategy="mean"), groups=[0, 1]))
    ])

    plot_model(pipe)
