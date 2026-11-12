import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn_compat.utils.estimator_checks import parametrize_with_checks

from sklego.mixture import GMMOutlierDetector
from sklego.preprocessing import OutlierRemover
from tests.conftest import GAUSSIAN_MIXTURE_ARRAY_API_REASON, expect_array_api_failure


def _expected_failed_checks(estimator: OutlierRemover) -> dict[str, str]:
    """Only the wrapped detector decides whether `array_api_dispatch` breaks, so apply the rule to it."""
    expect_failure = expect_array_api_failure(GAUSSIAN_MIXTURE_ARRAY_API_REASON, applies_to=(GMMOutlierDetector,))
    return expect_failure(estimator.outlier_detector)


@parametrize_with_checks(
    [
        OutlierRemover(outlier_detector=GMMOutlierDetector(), refit=True),
        OutlierRemover(outlier_detector=IsolationForest(), refit=True),
    ],
    expected_failed_checks=_expected_failed_checks,
)
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        # As this transformer removes samples, it is not standard for sure
        "check_transformer_general",
        "check_methods_sample_order_invariance",  # leads to out of index
        "check_methods_subset_invariance",  # leads to different shapes
        "check_transformer_data_not_an_array",  # hash only supports a few types
        "check_pipeline_consistency",  # Discussed in https://github.com/koaning/scikit-lego/issues/643
    }:
        pytest.skip("OutlierRemover is a TrainOnlyTransformer")
    check(estimator)


def test_no_outliers(mocker):
    mock_outlier_detector = mocker.Mock()
    mock_outlier_detector.fit.return_value = None
    mock_outlier_detector.predict.return_value = np.array([1, 1])
    mocker.patch("sklego.preprocessing.outlier_remover.clone").return_value = mock_outlier_detector

    outlier_remover = OutlierRemover(outlier_detector=mock_outlier_detector, refit=True)
    outlier_remover.fit(X=np.array([[1, 1], [2, 2]]))
    assert len(outlier_remover.transform_train(np.array([[1, 1], [2, 2]]))) == 2


def test_remove_outlier(mocker):
    mock_outlier_detector = mocker.Mock()
    mock_outlier_detector.fit.return_value = None
    mock_outlier_detector.predict.return_value = np.array([-1])
    mocker.patch("sklego.preprocessing.outlier_remover.clone").return_value = mock_outlier_detector

    outlier_remover = OutlierRemover(outlier_detector=mock_outlier_detector, refit=True)
    outlier_remover.fit(X=np.array([[5, 5]]))
    assert len(outlier_remover.transform_train(np.array([[0, 0]]))) == 0


def test_do_not_refit(mocker):
    mock_outlier_detector = mocker.Mock()
    mock_outlier_detector.fit.return_value = None
    mock_outlier_detector.predict.return_value = np.array([-1])
    mocker.patch("sklego.preprocessing.outlier_remover.clone").return_value = mock_outlier_detector

    outlier_remover = OutlierRemover(outlier_detector=mock_outlier_detector, refit=False)
    outlier_remover.fit(X=np.array([[5, 5]]))
    mock_outlier_detector.fit.assert_not_called()


@pytest.mark.parametrize("container", [np.array, pd.DataFrame])
@pytest.mark.parametrize("method", ["transform_train", "transform", "fit_transform"])
def test_prefitted_detector(container, method, mocker):
    X = container([[0.0], [0.1], [0.2], [0.3], [10.0]])
    detector = IsolationForest(random_state=0, contamination=0.2).fit(X)
    expected = X[detector.predict(X) != -1]
    fit_spy = mocker.spy(detector, "fit")
    remover = OutlierRemover(detector, refit=False)

    if method == "fit_transform":
        result = remover.fit_transform(X)
    else:
        assert remover.fit(X) is remover
        result = getattr(remover, method)(X)

    assert remover.estimator_ is detector
    fit_spy.assert_not_called()
    assert len(expected) < len(X)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("container", [np.array, pd.DataFrame])
def test_prefitted_detector_tracks_training_data(container, mocker):
    X = container([[0.0], [0.1], [0.2], [0.3], [10.0]])
    X_new = container([[0.1], [20.0]])
    detector = IsolationForest(random_state=0, contamination=0.2).fit(X)
    fit_spy = mocker.spy(detector, "fit")
    remover = OutlierRemover(detector, refit=False).fit(X)

    np.testing.assert_array_equal(remover.transform(X_new), X_new)
    remover.fit(X_new)
    np.testing.assert_array_equal(remover.transform(X_new), X_new[detector.predict(X_new) != -1])
    np.testing.assert_array_equal(remover.transform(X), X)
    fit_spy.assert_not_called()


def test_refit_clones_detector():
    X = np.array([[0.0], [0.1], [0.2], [0.3], [10.0]])
    detector = IsolationForest(random_state=0, contamination=0.2).fit(X)
    original_trees = detector.estimators_
    remover = OutlierRemover(detector, refit=True).fit(X)

    assert remover.estimator_ is not detector
    assert detector.estimators_ is original_trees
    np.testing.assert_array_equal(remover.transform(X), X[detector.predict(X) != -1])


def test_pipeline_integration():
    np.random.seed(42)
    dataset = np.concatenate([np.random.normal(0, 1, (2000, 2))])
    isolation_forest_remover = OutlierRemover(outlier_detector=IsolationForest())
    gmm_remover = OutlierRemover(outlier_detector=GMMOutlierDetector())
    pipeline = Pipeline(
        [
            ("isolation_forest_remover", isolation_forest_remover),
            ("gmm_remover", gmm_remover),
            ("kmeans", KMeans()),
        ]
    )
    pipeline.fit(dataset)
    pipeline.transform(dataset)
