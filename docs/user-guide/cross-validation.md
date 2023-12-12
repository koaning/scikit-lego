# Cross Validation

## TimeGapSplit

We allow for a timeseries split that contains a gap.

You won't always need it, but sometimes you consider these two situations;

- If you have multiple samples per timestamp: you want to make sure that a timestamp doesn’t appear at the same time in training and validation folds.
- If your target is looking $x$ days ahead in the future. In this case you cannot construct the target of the last $x$ days of your available data. It means that when you put your model in production, the first day that you are going to score is always $x$ days after your last training sample, therefore you should select the best model according to that setup.

    In other words, if you keep that gap in the validation, your metric might be overestimated because those first $x$ days might be easier to predict since they are closer to the training set. If you want to be strict in terms of robustness you might want to replicate in the CV exactly this real-world behaviour, and thus you want to introduce a gap of x days between your training and validation folds.

[`TimeGapSplit`][time-gap-split-api] provides 4 parameters to really reproduce your production implementation in your cross-validation schema. We will demonstrate this in a code example below.

### Examples

Let's make some random data to start with, and next define a plotting function.

```py
--8<-- "docs/_scripts/cross-validation.py:setup"
```

--8<-- "docs/_static/cross-validation/ts.md"

```py title="Example 1"
--8<-- "docs/_scripts/cross-validation.py:example-1"
```

![example-1](/_static/cross-validation/example-1.png)

```py title="Example 2"
--8<-- "docs/_scripts/cross-validation.py:example-2"
```

![example-2](/_static/cross-validation/example-2.png)

`window="expanding"` is the closest to scikit-learn implementation:

```py title="Example 3"
--8<-- "docs/_scripts/cross-validation.py:example-3"
```

![example-3](/_static/cross-validation/example-3.png)

If `train_duration` is not passed the training duration is the maximum without overlapping validation folds:

```py title="Example 4"
--8<-- "docs/_scripts/cross-validation.py:example-4"
```

![example-4](/_static/cross-validation/example-4.png)

If train and valid duration would lead to unwanted amounts of splits n_splits can set a maximal amount of splits

```py title="Example 5"
--8<-- "docs/_scripts/cross-validation.py:example-5"
```

![example-5](/_static/cross-validation/example-5.png)

```py title="Summary"
--8<-- "docs/_scripts/cross-validation.py:summary"
```

--8<-- "docs/_static/cross-validation/summary.md"

## GroupTimeSeriesSplit

In a time series problem it is possible that not every time unit (e.g. years) has the same amount of rows/observations.
This makes a normal kfold split impractical as you cannot specify a certain timeframe per fold (e.g. 5 years), because this can cause the folds' sizes to be very different.

With [`GroupTimeSeriesSplit`][group-ts-split-api] you can specify the amount of folds you want (e.g. `n_splits=3`) and `GroupTimeSeriesSplit` will calculate itself folds in such a way that the amount of observations per fold are as similar as possible.

The folds are created with a smartly modified brute forced method. This still means that for higher `n_splits` values in combination with many different unique time periods (e.g. 100 different years, thus 100 groups) the generation of the optimal split points can take minutes to hours.

!!! info
    `UserWarnings` are raised when `GroupTimeSeriesSplit` expects to be running over a minute. Of course, this actual runtime depends on your machine's specifications.

### Examples

First let's create an example data set:

```py
--8<-- "docs/_scripts/cross-validation.py:grp-setup"
```

--8<-- "docs/_static/cross-validation/grp-ts.md"

Create a `GroupTimeSeriesSplit` cross-validator with kfold/n_splits = 3:

```py
--8<-- "docs/_scripts/cross-validation.py:grp-ts-split"
```

```console
Fold 1:
Train = [2000, 2000, 2000, 2001]
Test = [2002, 2002, 2003]


Fold 2:
Train = [2002, 2002, 2003]
Test = [2004, 2004, 2004, 2004, 2004]


Fold 3:
Train = [2004, 2004, 2004, 2004, 2004]
Test = [2005, 2005, 2006, 2006, 2007]
```

![grp-ts-split](/_static/cross-validation/group-time-series-split.png)

As you can see above `GroupTimeSeriesSplit` keeps the order of the time chronological and makes sure that the same time value won't appear in both the train and test set of the same fold.

`GroupTimeSeriesSplit` also has the `.summary()` method, in which is shown which time values are grouped together. Because of the chronological order the train and test folds need to be, the amount of `groups` is always `n_splits` + 1. (see the four folds in the image above with `Kfold=3`)

```py title="Summary"
--8<-- "docs/_scripts/cross-validation.py:grp-summary"
```

--8<-- "docs/_static/cross-validation/grp-summary.md"

To use `GroupTimeSeriesSplit` with sklearn's [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html):

```py
--8<-- "docs/_scripts/cross-validation.py:grid-search"
```

[time-gap-split-api]: /api/model-selection#sklego.model_selection.TimeGapSplit
[group-ts-split-api]: /api/model-selection#sklego.model_selection.GroupTimeSeriesSplit
