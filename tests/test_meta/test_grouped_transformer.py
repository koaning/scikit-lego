import itertools as it

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, TargetEncoder
from sklearn.utils import check_X_y
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.datasets import load_heroes, load_penguins
from sklego.meta import GroupedTransformer
from tests.conftest import k_vals, n_vals, np_types


@parametrize_with_checks([GroupedTransformer(StandardScaler(), groups=0, check_X=True)])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        "check_transformer_data_not_an_array",  # TODO: Look into this
        "check_fit2d_1feature",  # custom message
    }:
        pytest.skip()

    check(estimator)


@pytest.fixture(scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)])
def dataset_with_single_grouping(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.normal(0, 2, (n,))

    X, y = check_X_y(X, y)

    # Make sure all groups are present
    groups = np.repeat([0, 1], repeats=(X.shape[0] + 1) // 2)[: X.shape[0], np.newaxis]
    X_with_groups = np.concatenate([groups, X], axis=1)
    grouper = 0  # First column

    return X, y, groups, X_with_groups, grouper


@pytest.fixture(scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)])
def dataset_with_multiple_grouping(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.normal(0, 2, (n,))

    X, y = check_X_y(X, y)
    # Make sure all groups are present
    groups = np.tile(
        # 4x2 array, repeated until it fits
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        reps=((X.shape[0] + 3) // 4, 1),
    )[: X.shape[0], :]
    X_with_groups = np.concatenate([groups, X], axis=1)

    grouper = (0, 1)

    return X, y, groups, X_with_groups, grouper


@pytest.fixture(scope="module")
def scaling_range():
    # Some weird interval to make sure we test the right values
    return (13, 42)


@pytest.fixture()
def multiple_obs_fitter():
    from sklearn.base import BaseEstimator, TransformerMixin

    class MultipleObsFitter(BaseEstimator, TransformerMixin):
        """A transformer that needs more than 1 value to fit"""

        def fit(self, X, y=None):
            if len(X) <= 1:
                raise ValueError("Need more than 1 value to fit")

            return self

        def transform(X):
            return X

    return MultipleObsFitter()


@pytest.fixture(scope="module")
def penguins_df():
    df = load_penguins(as_frame=True).dropna()
    X = df.drop(columns="species")

    return X


@pytest.fixture(scope="module")
def penguins(penguins_df):
    return penguins_df.values


def test_all_groups_scaled(dataset_with_single_grouping, scaling_range):
    X, y, groups, X_with_groups, grouper = dataset_with_single_grouping

    trf = MinMaxScaler(scaling_range)
    transformer = GroupedTransformer(trf, groups=grouper)
    transformed = transformer.fit(X_with_groups, y).transform(X_with_groups)

    df_with_groups = pd.concat([pd.Series(groups.flatten(), name="G"), pd.DataFrame(transformed)], axis=1)

    assert np.allclose(df_with_groups.groupby("G").min(), scaling_range[0])
    assert np.allclose(df_with_groups.groupby("G").max(), scaling_range[1])


def test_group_correlation_minmaxscaler(dataset_with_single_grouping, scaling_range):
    X, y, groups, X_with_groups, grouper = dataset_with_single_grouping

    trf = MinMaxScaler(scaling_range)
    transformer = GroupedTransformer(trf, groups=grouper)
    transformed = transformer.fit(X_with_groups, y).transform(X_with_groups)

    # For each column, check that all grouped correlations are 1 (because MinMaxScaler scales linear)
    for col in range(X.shape[1]):
        assert (
            pd.concat(
                [
                    pd.Series(groups.flatten(), name="group"),
                    pd.Series(X[:, col], name="original"),
                    pd.Series(transformed[:, col], name="transformed"),
                ],
                axis=1,
            )
            .groupby("group")
            .corr()
            .pipe(np.allclose, 1)
        )


def test_set_params():
    trf = StandardScaler(with_std=False)
    transformer = GroupedTransformer(trf, groups=0)

    transformer.set_params(transformer__with_std=True)
    assert trf.with_std


def test_get_params():
    trf = StandardScaler(with_std=False)
    transformer = GroupedTransformer(trf, groups=0)

    assert transformer.get_params() == {
        "transformer__with_mean": True,
        "transformer__with_std": False,
        "transformer__copy": True,
        "transformer": trf,
        "groups": 0,
        "use_global_model": True,
        "check_X": True,
    }


def test_non_transformer(dataset_with_single_grouping):
    X, y, _, _, grouper = dataset_with_single_grouping

    # This is not a transformer
    trf = LinearRegression()
    transformer = GroupedTransformer(trf, groups=grouper)

    with pytest.raises(ValueError):
        transformer.fit(X, y)


def test_multiple_grouping_columns(dataset_with_multiple_grouping, scaling_range):
    X, y, groups, X_with_groups, grouper = dataset_with_multiple_grouping

    trf = MinMaxScaler(scaling_range)
    transformer = GroupedTransformer(trf, groups=grouper)
    transformed = transformer.fit(X_with_groups, y).transform(X_with_groups)

    df_with_groups = pd.concat([pd.DataFrame(groups, columns=["A", "B"]), pd.DataFrame(transformed)], axis=1)

    assert np.allclose(df_with_groups.groupby(["A", "B"]).min(), scaling_range[0])

    # If a group has a single element, it defaults to min, so check whether all maxes are one of the bounds
    maxes = df_with_groups.groupby(["A", "B"]).max()
    assert np.all(
        np.isclose(maxes, scaling_range[1]) | np.isclose(maxes, scaling_range[0])
        # We have at least some groups larger than 1, so there we should find the max of the range
    ) and np.any(np.isclose(maxes, scaling_range[1]))


def test_missing_groups_transform_global(dataset_with_single_grouping, scaling_range):
    X, y, groups, X_with_groups, grouper = dataset_with_single_grouping

    trf = MinMaxScaler(scaling_range)
    transformer = GroupedTransformer(trf, groups=grouper)
    transformer.fit(X_with_groups, y)

    # Array with 2 rows, first column a new group. Remaining top are out of range so should be the range
    X_test = np.concatenate([np.array([[3], [3]]), np.stack([X.min(axis=0), X.max(axis=0)], axis=0)], axis=1)

    transformed = transformer.transform(X_test)

    # Top row should all be equal to the small value of the range, bottom the other
    assert np.allclose(transformed[0, :], scaling_range[0])
    assert np.allclose(transformed[1, :], scaling_range[1])


def test_missing_groups_transform_noglobal(dataset_with_single_grouping, scaling_range):
    X, y, groups, X_with_groups, grouper = dataset_with_single_grouping

    trf = MinMaxScaler(scaling_range)
    transformer = GroupedTransformer(trf, groups=grouper, use_global_model=False)
    transformer.fit(X_with_groups, y)

    # Array with 2 rows, first column a new group. Remaining top are out of range so should be the range
    X_test = np.concatenate([np.array([[3], [3]]), np.stack([X.min(axis=0) - 1, X.max(axis=0) + 1], axis=0)], axis=1)

    with pytest.raises(ValueError):
        transformer.transform(X_test)


def test_exception_in_group(multiple_obs_fitter):
    X = np.array(
        [
            [1, 2],
            [1, 0],
            [2, 1],
        ]
    )

    # Only works on groups greater than 1, so will raise an error in group 2
    transformer = GroupedTransformer(multiple_obs_fitter, groups=0, use_global_model=False)

    with pytest.raises(ValueError) as e:
        transformer.fit(X)

        assert "group 2" in str(e)


def test_array_with_strings():
    X = np.array(
        [
            ("group0", 2),
            ("group0", 0),
            ("group1", 1),
            ("group1", 3),
        ],
        dtype="object",
    )

    trf = MinMaxScaler()
    transformer = GroupedTransformer(trf, groups=0, use_global_model=False)
    transformer.fit_transform(X)


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame])
def test_df(penguins_df, frame_func):
    penguins_df = frame_func(penguins_df)
    meta = GroupedTransformer(StandardScaler(), groups=["island", "sex"])

    transformed = meta.fit_transform(penguins_df)

    # 2 columns for grouping not in the result
    assert transformed.shape == (penguins_df.shape[0], penguins_df.shape[1] - 2)


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame])
def test_df_missing_group(penguins_df, frame_func):
    meta = GroupedTransformer(StandardScaler(), groups=["island", "sex"])

    # Otherwise the fixture is changed
    X = penguins_df.copy()
    X.loc[0, "island"] = None
    X = frame_func(X)
    with pytest.raises(ValueError):
        meta.fit_transform(X)


def test_array_with_multiple_string_cols(penguins):
    X = penguins

    # BROKEN: Failing due to negative indexing... kind of an edge case
    meta = GroupedTransformer(StandardScaler(), groups=[0, -1])

    transformed = meta.fit_transform(X)

    # 2 columns for grouping not in the result
    assert transformed.shape == (X.shape[0], X.shape[1] - 2)


def test_grouping_column_not_in_array(penguins):
    X = penguins

    meta = GroupedTransformer(StandardScaler(), groups=[0, 5])

    # This should raise ValueError
    with pytest.raises(ValueError):
        meta.fit_transform(X[:, :3])


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame])
def test_grouping_column_not_in_df(penguins_df, frame_func):
    meta = GroupedTransformer(StandardScaler(), groups=["island", "unexisting_column"])

    # This should raise ValueError
    with pytest.raises(ValueError):
        meta.fit_transform(frame_func(penguins_df))


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame])
def test_no_grouping(penguins_df, frame_func):
    penguins_numeric = frame_func(penguins_df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])

    meta = GroupedTransformer(StandardScaler(), groups=None)
    nonmeta = StandardScaler()

    assert (meta.fit_transform(penguins_numeric) == nonmeta.fit_transform(penguins_numeric)).all()


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame])
def test_with_y(penguins_df, frame_func):
    X = frame_func(penguins_df.drop(columns=["sex"]))
    y = penguins_df["sex"]

    meta = GroupedTransformer(StandardScaler(), groups="island")

    # This should work fine
    transformed = meta.fit_transform(X, y)

    # 1 column for grouping not in the result
    assert transformed.shape == (X.shape[0], X.shape[1] - 1)


@pytest.mark.parametrize(
    "transformer",
    (
        TargetEncoder(target_type="continuous", random_state=123),
        Pipeline(
            [("encoder", TargetEncoder(target_type="continuous", random_state=123)), ("scaler", StandardScaler())]
        ),
        Pipeline(
            [("scaler", StandardScaler()), ("encoder", TargetEncoder(target_type="continuous", random_state=123))]
        ),
    ),
)
def test_transform_with_y(transformer):
    """Test that the GroupedTransformer works with a transformer that requires y."""

    df_heroes = (
        load_heroes(as_frame=True)
        .drop(columns="name")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .assign(role=lambda t: LabelEncoder().fit_transform(t["role"]))
    )

    target, groups, features = "attack", "attack_type", ["role"]
    X, y = df_heroes.drop(columns=target), df_heroes[target]

    is_melee = X[groups] == "Melee"

    X_melee = (
        clone(transformer)
        .set_output(transform="pandas")
        .fit(X.loc[is_melee, features], y.loc[is_melee])
        .transform(X.loc[is_melee, features])
    )
    X_ranged = (
        clone(transformer)
        .set_output(transform="pandas")
        .fit(X.loc[~is_melee, features], y.loc[~is_melee])
        .transform(X.loc[~is_melee, features])
    )

    X_naive = pd.concat([X_melee, X_ranged]).sort_index()
    X_grouped = (
        GroupedTransformer(clone(transformer), groups=groups, check_X=False)
        .fit(X.loc[:, [groups] + features], y)
        .transform(X.loc[:, [groups] + features])
    )

    assert np.allclose(X_naive.to_numpy(), X_grouped)
