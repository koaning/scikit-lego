from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.datasets import load_chicken
from sklego.meta import GroupedClassifier, GroupedPredictor, GroupedRegressor


@parametrize_with_checks(
    [
        meta_cls(estimator=LinearRegression(), groups=0, use_global_model=True)
        for meta_cls in [GroupedPredictor, GroupedRegressor]
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        "check_no_attributes_set_in_init",  # Setting **shrinkage_kwargs in init
        "check_estimators_empty_data_messages",  # Custom message
        "check_fit2d_1feature",  # Custom message (after grouping we are left with zero features)
        "check_supervised_y_2d",  # Unsure about this
    }:
        pytest.skip()

    if check.func.__name__ == "check_regressors_train" and estimator.__class__ is GroupedPredictor:
        # Can't use `isinstance(estimator, GroupedPredictor)` since that's true for both cases
        pytest.skip()

    check(estimator)


@pytest.fixture
def random_xy_grouped_clf_different_classes(request):
    group_size = request.param.get("group_size")
    y_choices_grpa = request.param.get("y_choices_grpa")
    y_choices_grpb = request.param.get("y_choices_grpb")

    np.random.seed(43)
    group_col = np.repeat(["A", "B"], group_size)
    x_col = np.random.normal(size=group_size * 2)
    y_col = np.hstack(
        [
            np.random.choice(y_choices_grpa, size=group_size),
            np.random.choice(y_choices_grpb, size=group_size),
        ]
    )
    df = pd.DataFrame({"group": group_col, "x": x_col, "y": y_col})
    return df


def test_chickweight_df1_keys():
    df = load_chicken(as_frame=True)
    mod = GroupedPredictor(estimator=LinearRegression(), groups="diet")
    mod.fit(df[["time", "diet"]], df["weight"])
    assert set(mod.estimators_.keys()) == {1, 2, 3, 4}


def test_chickweight_df2_keys():
    df = load_chicken(as_frame=True)
    mod = GroupedPredictor(estimator=LinearRegression(), groups="chick")
    mod.fit(df[["time", "chick"]], df["weight"])
    assert set(mod.estimators_.keys()) == set(range(1, 50 + 1))


def test_chickweight_can_do_fallback():
    df = load_chicken(as_frame=True)
    mod = GroupedPredictor(estimator=LinearRegression(), groups="diet")
    mod.fit(df[["time", "diet"]], df["weight"])
    assert set(mod.estimators_.keys()) == {1, 2, 3, 4}
    to_predict = pd.DataFrame({"time": [21, 21], "diet": [5, 6]})
    assert mod.predict(to_predict).shape == (2,)
    assert mod.predict(to_predict)[0] == mod.predict(to_predict)[1]


def test_chickweight_can_do_fallback_proba():
    df = load_chicken(as_frame=True)
    y = np.where(df.weight > df.weight.mean(), 1, 0)
    mod = GroupedPredictor(estimator=LogisticRegression(), groups="diet")
    mod.fit(df[["time", "diet"]], y)
    assert set(mod.estimators_.keys()) == {1, 2, 3, 4}
    to_predict = pd.DataFrame({"time": [21, 21], "diet": [5, 6]})
    assert mod.predict_proba(to_predict).shape == (2, 2)
    assert (mod.predict_proba(to_predict)[0] == mod.predict_proba(to_predict)[1]).all()


@pytest.mark.parametrize(
    "random_xy_grouped_clf_different_classes",
    [
        {"group_size": 10, "y_choices_grpa": [0, 1, 2], "y_choices_grpb": [0, 1, 2, 4]},
        {"group_size": 10, "y_choices_grpa": [0, 2], "y_choices_grpb": [0, 2]},
        {"group_size": 10, "y_choices_grpa": [0, 1, 2, 3], "y_choices_grpb": [0, 4]},
        {"group_size": 10, "y_choices_grpa": [0, 1, 2], "y_choices_grpb": [0, 3]},
    ],
    indirect=True,
)
def test_predict_proba_has_same_columns_as_distinct_labels(
    random_xy_grouped_clf_different_classes,
):
    mod = GroupedPredictor(estimator=LogisticRegression(), groups="group")
    X, y = (
        random_xy_grouped_clf_different_classes[["group", "x"]],
        random_xy_grouped_clf_different_classes["y"],
    )
    _ = mod.fit(X, y)
    y_proba = mod.predict_proba(X)

    # Ensure the number of col output is always equal to the cardinality of the labels
    assert len(random_xy_grouped_clf_different_classes["y"].unique()) == y_proba.shape[1]


@pytest.mark.parametrize(
    "random_xy_grouped_clf_different_classes",
    [
        {"group_size": 5, "y_choices_grpa": [0, 1, 2], "y_choices_grpb": [0, 2]},
    ],
    indirect=True,
)
def test_predict_proba_correct_zeros_same_and_different_labels(
    random_xy_grouped_clf_different_classes,
):
    mod = GroupedPredictor(estimator=LogisticRegression(), groups="group")

    X, y = (
        random_xy_grouped_clf_different_classes[["group", "x"]],
        random_xy_grouped_clf_different_classes["y"],
    )
    _ = mod.fit(X, y)
    y_proba = mod.predict_proba(X)

    df_proba = pd.concat(
        [random_xy_grouped_clf_different_classes["group"], pd.DataFrame(y_proba)],
        axis=1,
    )

    # Take distinct labels for group A and group B
    labels_a, labels_b = random_xy_grouped_clf_different_classes.groupby("group").agg({"y": set}).sort_index()["y"]

    # Ensure for the common labels there are no zeros
    in_common_labels = labels_a.intersection(labels_b)
    assert all((df_proba.loc[:, label] != 0).all() for label in in_common_labels)

    # Ensure for the non common labels there are only zeros
    label_not_in_group = {
        "A": list(labels_b.difference(labels_a)),
        "B": list(labels_a.difference(labels_b)),
    }
    for grp_name, grp in df_proba.groupby("group"):
        assert all((grp.loc[:, label] == 0).all() for label in label_not_in_group[grp_name])


def test_fallback_can_raise_error():
    df = load_chicken(as_frame=True)
    mod = GroupedPredictor(
        estimator=LinearRegression(),
        groups="diet",
        use_global_model=False,
        shrinkage=None,
    )
    mod.fit(df[["time", "diet"]], df["weight"])
    to_predict = pd.DataFrame({"time": [21, 21], "diet": [5, 6]})
    with pytest.raises(ValueError) as e:
        mod.predict(to_predict)
        assert "found a group" in str(e)


def test_chickweight_raise_error_group_col_missing():
    df = load_chicken(as_frame=True)
    mod = GroupedPredictor(estimator=LinearRegression(), groups="diet")
    mod.fit(df[["time", "diet"]], df["weight"])
    with pytest.raises(ValueError) as e:
        mod.predict(df[["time", "chick"]])
        assert "not in columns" in str(e)


def test_chickweight_raise_error_value_col_missing():
    df = load_chicken(as_frame=True)
    mod = GroupedPredictor(estimator=LinearRegression(), groups="diet")
    mod.fit(df[["time", "diet"]], df["weight"])

    with pytest.raises(ValueError):
        # Former test not valid anymore because we don't check for value columns
        # mod.predict(df[["diet", "chick"]])
        mod.predict(df[["diet"]])


def test_chickweight_np_keys():
    df = load_chicken(as_frame=True)
    mod = GroupedPredictor(estimator=LinearRegression(), groups=[1, 2])
    mod.fit(df[["time", "chick", "diet"]].values, df["weight"].values)
    # there should still only be 50 groups on this dataset
    assert len(mod.estimators_.keys()) == 50


def test_chickweigt_string_groups():
    df = load_chicken(as_frame=True)
    df["diet"] = ["omgomgomg" + s for s in df["diet"].astype(str)]

    X = df[["time", "diet"]]
    X_np = np.array(X)

    y = df["weight"]

    # This should NOT raise errors
    GroupedPredictor(LinearRegression(), groups=["diet"]).fit(X, y).predict(X)
    GroupedPredictor(LinearRegression(), groups=1).fit(X_np, y).predict(X_np)


@pytest.fixture
def shrinkage_data():
    df = pd.DataFrame(
        {
            "Planet": ["Earth", "Earth", "Earth", "Earth"],
            "Country": ["NL", "NL", "BE", "BE"],
            "City": ["Amsterdam", "Rotterdam", "Antwerp", "Brussels"],
            "Target": [1, 3, 2, 4],
        }
    )

    means = {
        "Earth": 2.5,
        "NL": 2,
        "BE": 3,
        "Amsterdam": 1,
        "Rotterdam": 3,
        "Antwerp": 2,
        "Brussels": 4,
    }

    return df, means


def test_constant_shrinkage(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    shrink_est = GroupedPredictor(
        DummyRegressor(),
        ["Planet", "Country", "City"],
        shrinkage="constant",
        use_global_model=False,
        alpha=0.1,
    )

    shrinkage_factors = np.array([0.01, 0.09, 0.9])

    shrink_est.fit(X, y)

    expected_prediction = [
        np.array([means["Earth"], means["NL"], means["Amsterdam"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["NL"], means["Rotterdam"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"], means["Antwerp"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"], means["Brussels"]]) @ shrinkage_factors,
    ]

    for exp, pred in zip(expected_prediction, shrink_est.predict(X).tolist()):
        assert pytest.approx(exp) == pred


def test_relative_shrinkage(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    shrink_est = GroupedPredictor(
        DummyRegressor(),
        ["Planet", "Country", "City"],
        shrinkage="relative",
        use_global_model=False,
    )

    shrinkage_factors = np.array([4, 2, 1]) / 7

    shrink_est.fit(X, y)

    expected_prediction = [
        np.array([means["Earth"], means["NL"], means["Amsterdam"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["NL"], means["Rotterdam"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"], means["Antwerp"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"], means["Brussels"]]) @ shrinkage_factors,
    ]

    for exp, pred in zip(expected_prediction, shrink_est.predict(X).tolist()):
        assert pytest.approx(exp) == pred


def test_min_n_obs_shrinkage(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    shrink_est = GroupedPredictor(
        DummyRegressor(),
        ["Planet", "Country", "City"],
        shrinkage="min_n_obs",
        use_global_model=False,
        min_n_obs=2,
    )

    shrink_est.fit(X, y)

    expected_prediction = [means["NL"], means["NL"], means["BE"], means["BE"]]

    for exp, pred in zip(expected_prediction, shrink_est.predict(X).tolist()):
        assert pytest.approx(exp) == pred


def test_min_n_obs_shrinkage_too_little_obs(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    too_big_n_obs = X.shape[0] + 1

    shrink_est = GroupedPredictor(
        DummyRegressor(),
        ["Planet", "Country", "City"],
        shrinkage="min_n_obs",
        use_global_model=False,
        min_n_obs=too_big_n_obs,
    )

    with pytest.raises(ValueError) as e:
        shrink_est.fit(X, y)

        assert f"There is no group with size greater than or equal to {too_big_n_obs}" in str(e)


def test_custom_shrinkage(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    def shrinkage_func(group_sizes):
        n = len(group_sizes)
        return np.repeat(1 / n, n)

    shrink_est = GroupedPredictor(
        DummyRegressor(),
        ["Planet", "Country", "City"],
        shrinkage=shrinkage_func,
        use_global_model=False,
    )

    shrinkage_factors = np.array([1, 1, 1]) / 3

    shrink_est.fit(X, y)

    expected_prediction = [
        np.array([means["Earth"], means["NL"], means["Amsterdam"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["NL"], means["Rotterdam"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"], means["Antwerp"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"], means["Brussels"]]) @ shrinkage_factors,
    ]

    assert expected_prediction == shrink_est.predict(X).tolist()


def test_custom_shrinkage_wrong_return_type(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    def shrinkage_func(group_sizes):
        return group_sizes

    with pytest.raises(ValueError) as e:
        shrink_est = GroupedPredictor(
            DummyRegressor(),
            ["Planet", "Country", "City"],
            shrinkage=shrinkage_func,
            use_global_model=False,
        )

        shrink_est.fit(X, y)

        assert "should return an np.ndarray" in str(e)


def test_custom_shrinkage_wrong_length(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    def shrinkage_func(group_sizes):
        n = len(group_sizes)
        return np.repeat(1 / n, n + 1)

    with pytest.raises(ValueError) as e:
        shrink_est = GroupedPredictor(
            DummyRegressor(),
            ["Planet", "Country", "City"],
            shrinkage=shrinkage_func,
            use_global_model=False,
        )

        shrink_est.fit(X, y)

        assert ".shape should be " in str(e)


def test_custom_shrinkage_raises_error(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    def shrinkage_func(group_sizes):
        raise KeyError("This function is bad and you should feel bad")

    with pytest.raises(ValueError) as e:
        shrink_est = GroupedPredictor(
            DummyRegressor(),
            ["Planet", "Country", "City"],
            shrinkage=shrinkage_func,
            use_global_model=False,
        )

        shrink_est.fit(X, y)

        assert "you should feel bad" in str(e) and "while checking the shrinkage function" in str(e)


@pytest.mark.parametrize("wrong_func", [list(), tuple(), dict(), 9])
def test_invalid_shrinkage(shrinkage_data, wrong_func):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    with pytest.raises(ValueError) as e:
        shrink_est = GroupedPredictor(
            DummyRegressor(),
            ["Planet", "Country", "City"],
            shrinkage=wrong_func,
            use_global_model=False,
        )

        shrink_est.fit(X, y)

        assert "Invalid shrinkage specified." in str(e)


def test_global_model_shrinkage(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    shrink_est_without_global = GroupedPredictor(
        DummyRegressor(),
        ["Planet", "Country", "City"],
        shrinkage="min_n_obs",
        use_global_model=False,
        min_n_obs=2,
    )

    shrink_est_with_global = GroupedPredictor(
        DummyRegressor(),
        ["Country", "City"],
        shrinkage="min_n_obs",
        use_global_model=True,
        min_n_obs=2,
    )

    shrink_est_without_global.fit(X, y)
    # Drop planet because otherwise it is seen as a value column
    shrink_est_with_global.fit(X.drop(columns="Planet"), y)

    pd.testing.assert_series_equal(
        shrink_est_with_global.predict(X.drop(columns="Planet")),
        shrink_est_without_global.predict(X),
    )


def test_shrinkage_single_group(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    shrink_est = GroupedPredictor(
        DummyRegressor(),
        "Country",
        shrinkage="constant",
        use_global_model=True,
        alpha=0.1,
    )

    shrinkage_factors = np.array([0.1, 0.9])

    # Drop planet and city because otherwise they are seen as value columns
    shrink_est.fit(X[["Country"]], y)

    expected_prediction = [
        np.array([means["Earth"], means["NL"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["NL"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"]]) @ shrinkage_factors,
    ]

    assert expected_prediction == shrink_est.predict(X[["Country"]]).tolist()


def test_shrinkage_single_group_no_global(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    with pytest.raises(ValueError) as e:
        shrink_est = GroupedPredictor(
            DummyRegressor(),
            "Country",
            shrinkage="constant",
            use_global_model=False,
            alpha=0.1,
        )
        shrink_est.fit(X, y)

        assert "Cannot do shrinkage with a single group if use_global_model is False" in str(e)


def test_unexisting_shrinkage_func(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    with pytest.raises(ValueError) as e:
        unexisting_func = "some_highly_unlikely_function_name"

        shrink_est = GroupedPredictor(
            estimator=DummyRegressor(),
            groups=["Planet", "Country"],
            shrinkage=unexisting_func,
        )

        shrink_est.fit(X, y)

        assert "shrinkage function" in str(e)


def test_unseen_groups_shrinkage(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    shrink_est = GroupedPredictor(DummyRegressor(), ["Planet", "Country", "City"], shrinkage="constant", alpha=0.1)

    shrink_est.fit(X, y)

    unseen_group = pd.DataFrame({"Planet": ["Earth"], "Country": ["DE"], "City": ["Hamburg"]})

    with pytest.raises(ValueError) as e:
        shrink_est.predict(X=pd.concat([unseen_group] * 4, axis=0))
        assert "found a group" in str(e)


def test_predict_missing_group_column(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df["Target"]

    shrink_est = GroupedPredictor(
        DummyRegressor(),
        ["Planet", "Country", "City"],
        shrinkage="constant",
        use_global_model=False,
        alpha=0.1,
    )

    shrink_est.fit(X, y)

    with pytest.raises(ValueError) as e:
        shrink_est.predict(X.drop(columns=["Country"]))
        assert "group columns" in str(e)


def test_predict_missing_value_column(shrinkage_data):
    df, means = shrinkage_data

    value_column = "predictor"

    X, y = df.drop(columns="Target"), df["Target"]
    X = X.assign(**{value_column: np.random.normal(size=X.shape[0])})

    shrink_est = GroupedPredictor(
        LinearRegression(),
        ["Planet", "Country", "City"],
        shrinkage="constant",
        use_global_model=False,
        alpha=0.1,
    )

    shrink_est.fit(X, y)

    with pytest.raises(ValueError) as e:
        shrink_est.predict(X.drop(columns=[value_column]))
        assert "columns to use" in str(e)


def test_bad_shrinkage_value_error():
    with pytest.raises(ValueError) as e:
        df = load_chicken(as_frame=True)
        mod = GroupedPredictor(estimator=LinearRegression(), groups="diet", shrinkage="dinosaurhead")
        mod.fit(df[["time", "diet"]], df["weight"])
        assert "shrinkage function" in str(e)


def test_missing_check():
    df = load_chicken(as_frame=True)

    X, y = df.drop(columns="weight"), df["weight"]
    # create missing value
    X.loc[0, "chick"] = np.nan
    model = make_pipeline(SimpleImputer(), LinearRegression())

    # Should not raise error, check is disabled
    m = GroupedPredictor(model, groups=["diet"], check_X=False).fit(X, y)
    m.predict(X)

    # Should raise error, check is still enabled
    with pytest.raises(ValueError) as e:
        GroupedPredictor(model, groups=["diet"]).fit(X, y)
        assert "contains NaN" in str(e)


def test_has_decision_function():
    # needed as for example cross_val_score(pipe, X, y, cv=5, scoring="roc_auc", error_score='raise') may fail
    # otherwise, see https://github.com/koaning/scikit-lego/issues/511
    df = load_chicken(as_frame=True)

    X, y = df.drop(columns="weight"), df["weight"]
    # This should NOT raise errors
    GroupedPredictor(LogisticRegression(max_iter=2000), groups=["diet"]).fit(X, y).decision_function(X)


@pytest.mark.parametrize(
    "meta_cls,estimator,context",
    [
        (GroupedRegressor, LinearRegression(), does_not_raise()),
        (GroupedClassifier, LogisticRegression(), does_not_raise()),
        (GroupedRegressor, LogisticRegression(), pytest.raises(ValueError)),
        (GroupedClassifier, LinearRegression(), pytest.raises(ValueError)),
    ],
)
def test_specialized_classes(meta_cls, estimator, context):
    df = load_chicken(as_frame=True)
    with context:
        meta_cls(estimator=estimator, groups="diet").fit(df[["time", "diet"]], df["weight"].astype(int))


@pytest.mark.parametrize(
    "shrinkage,context",
    [
        (None, does_not_raise()),
        ("constant", pytest.raises(ValueError)),
        ("relative", pytest.raises(ValueError)),
        ("min_n_obs", pytest.raises(ValueError)),
        (lambda x: x, pytest.raises(ValueError)),
    ],
)
def test_clf_shrinkage(shrinkage, context):
    df = load_chicken(as_frame=True)
    with context:
        GroupedPredictor(estimator=LogisticRegression(), groups="diet", shrinkage=shrinkage).fit(
            df[["time", "diet"]], df["weight"].astype(int)
        )
