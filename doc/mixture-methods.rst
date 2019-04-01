Mixture Methods
===============

Gaussian Mixture Models (GMMs) are flexible building blocks for other
machine learning algorithms. This is in part because they are
great approximations for general probability distributions but
also because they remain somewhat interpretable even when the
dataset gets very complex. This package makes use of GMMs to construct
other algorithms.

Classification
--------------

Below is some example code of how you might use a GMM
from sklego to perform classification.

.. testcode:: python

    import numpy as np
    import matplotlib.pylab as plt

    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    from sklego.mixture import GMMClassifier

    n = 1000
    X, y = make_moons(n)
    X = X + np.random.normal(n, 0.12, (n, 2))
    X = StandardScaler().fit_transform(X)
    U = np.random.uniform(-2, 2, (10000, 2))

    mod = GMMClassifier(n_components=4).fit(X, y)

    plt.figure(figsize=(14, 3))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=mod.predict(X), s=8)
    plt.title("classes of points");

    plt.subplot(122)
    plt.scatter(U[:, 0], U[:, 1], c=mod.predict_proba(U)[:, 1], s=8)
    plt.title("classifier boundary");


.. image:: _static/outlier-clf.png

Outlier Detection
-----------------

Below is some example code of how you might use a GMM
from sklego to do outlier detection.

.. testcode:: python

    import numpy as np
    import matplotlib.pylab as plt

    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    from sklego.mixture import GMMOutlierDetector

    n = 1000
    X = make_moons(n)[0] + np.random.normal(n, 0.12, (n, 2))
    X = StandardScaler().fit_transform(X)
    U = np.random.uniform(-2, 2, (10000, 2))

    mod = GMMOutlierDetector(n_components=16, threshold=0.95).fit(X)

    plt.figure(figsize=(14, 3))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=mod.score_samples(X), s=8)
    plt.title("likelihood of points given mixture of 16 gaussians");

    plt.subplot(122)
    plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8)
    plt.title("outlier selection")

.. image:: _static/outlier-mixture.png