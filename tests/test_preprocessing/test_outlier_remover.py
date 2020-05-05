import pytest
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_checks

from sklego.common import flatten
from sklego.preprocessing import OutlierRemover
from sklego.mixture import GMMOutlierDetector


@pytest.mark.parametrize(
    "test_fn",
    flatten(
        [
            estimator_checks.check_transformers_unfitted,
            estimator_checks.check_fit2d_predict1d,
            estimator_checks.check_fit2d_1sample,
            estimator_checks.check_fit2d_1feature,
            estimator_checks.check_fit1d,
            estimator_checks.check_get_params_invariance,
            estimator_checks.check_set_params,
            estimator_checks.check_dont_overwrite_parameters,
            estimator_checks.check_transformers_unfitted,
        ]
    ),
)
def test_estimator_checks(test_fn):
    gmm_remover = OutlierRemover(outlier_detector=GMMOutlierDetector(), refit=True)
    test_fn(OutlierRemover.__name__, gmm_remover)

    isolation_forest_remover = OutlierRemover(
        outlier_detector=IsolationForest(), refit=True
    )
    test_fn(OutlierRemover.__name__, isolation_forest_remover)


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

    outlier_remover = OutlierRemover(
        outlier_detector=mock_outlier_detector, refit=False
    )
    outlier_remover.fit(X=np.array([[5, 5]]))
    mock_outlier_detector.fit.assert_not_called()


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
