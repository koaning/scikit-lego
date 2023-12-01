# Outliers

This package offers a few algorithms that might help you find outliers. Note that we offer a subset of algorithms that we could not find elsewhere.
If you're interested in more algorithms we might recommend you have a look at [pyod][pyod-docs] too. That said, we'll demonstrate a few approaches here.

## Decomposition Based Detection

The scikit-learn ecosystem offers many tools for dimensionality reduction. Two popular variants are [PCA][pca-api] and [UMAP][umap-api]. What is nice about both of these methods is that they can reduce the data but also apply the inverse operation.

![decomposition](/_static/outliers/decomposition.png)

This is similar to what an autoencoder might do. But let's now say that we have a dataset $X$ and that we're happy with our dimensionality reduction technique.
In this situation there's a balance between reduction of data and loss of information.

Suppose that we have a datapoint $x_{\text{orig}}$ we pass through our transformer after which we try to reconstruct it again. If $x_{\text{orig}}$ differs a lot from $x_{\text{reconstruct}}$ then we may have a good candidate to investigate as an outlier.

We'll demonstrate both methods briefly below, using this following function to make some plots.

!!! example "Data and functionalities"
    ```py
    --8<-- "docs/_scripts/outliers.py:setup"
    ```

### PCA Demonstration

Let's start with PCA methods to decompose and reconstruct the data, wrapped in the class [`PCAOutlierDetection`][pca-outlier-api].

```py
--8<-- "docs/_scripts/outliers.py:pca-outlier"
```

![pca-outlier](/_static/outliers/pca-outlier.png)

### UMAP Demonstration

Let's now do the same with UMAP, wrapped in the class [`UMAPOutlierDetection`][umap-outlier-api].

```py
--8<-- "docs/_scripts/outliers.py:umap-outlier"
```

![umap-outlier](/_static/outliers/umap-outlier.png)

One thing to keep in mind here: UMAP is _a lot slower_.

### Interpretation of Hyperparams

Both methods have a `n_components` and `threshold` parameter. The former tells the underlying transformer how many components to reduce to while the latter tells the model when to consider a reconstruction error "too big" for a datapoint not to be an outlier.

If the relative error is larger than the set threshold it will be detected as an outlier. Typically that means that the threshold will be a lower value between 0.0 and 0.1. You can also specify an `absolute` threshold if that is preferable.

The other parameters in both models are unique to their underlying transformer method.

## Density Based Detection

We've also got a few outlier detection techniques that are density based approaches. You will find a subset documented in the [mixture method section](/user-guide/mixture-methods) but for completeness we will also list them below here as a comparison.

### [GMMOutlierDetector][gmm-outlier-api] Demonstration

```py
--8<-- "docs/_scripts/outliers.py:gmm-outlier"
```

![gmm-outlier](/_static/outliers/gmm-outlier.png)

### [BayesianGMMOutlierDetector][bayesian-gmm-outlier-api] Demonstration

```py
--8<-- "docs/_scripts/outliers.py:bayesian-gmm-outlier"
```

![bayesian-gmm-outlier](/_static/outliers/bayesian-gmm-outlier.png)

Note that for these density based approaches the threshold needs to be interpreted differently. If you're interested, you can find more information [here](/user-guide/mixture-methods#detection-details).

## Model Based Outlier Detection

Suppose that you've got an accurate model. Then you could argue that when a datapoint disagrees with your model that it might be an outlier.

This library offers meta models that wrap estimators in order to become outlier detection models.

### Regression Based

If you have a regression model then we offer a [`RegressionOutlierDetector`][regr-outlier-api]. This model takes the output of the regression model and compares it against the true regression labels. If the difference between the label and predicted value is larger than a threshold then we output an outlier flag.

Note that in order to be complaint to the scikit-learn API we require that the `y`-label for the regression to be part of the `X` dataset.

```py
--8<-- "docs/_scripts/outliers.py:regr-outlier"
```

![regr-outlier](/_static/outliers/regr-outlier.png)

[pca-outlier-api]: /api/decomposition#sklego.decomposition.pca_reconstruction.PCAOutlierDetection
[umap-outlier-api]: /api/decomposition#sklego.decomposition.umap_reconstruction.UMAPOutlierDetection.md
[gmm-outlier-api]: /api/mixture#sklego.mixture.gmm_outlier_detector.GMMOutlierDetector
[bayesian-gmm-outlier-api]: /api/mixture#sklego.mixture.bayesian_gmm_detector.BayesianGMMOutlierDetector
[regr-outlier-api]: /api/meta#sklego.meta.regression_outlier_detector.RegressionOutlierDetector

[pyod-docs]: https://pyod.readthedocs.io/en/latest/
[pca-api]: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
[umap-api]: https://umap-learn.readthedocs.io/en/latest/api.html#umap
