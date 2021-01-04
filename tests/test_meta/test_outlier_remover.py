import pytest
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from sklego.common import flatten
from sklego.meta import OutlierRemover
from sklego.mixture import GMMOutlierDetector

from tests.conftest import general_checks, select_tests


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_methods_subset_invariance"
        ]
    )
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
    mocker.patch("sklego.meta.outlier_remover.clone").return_value = mock_outlier_detector

    outlier_remover = OutlierRemover(outlier_detector=mock_outlier_detector, refit=True)
    outlier_remover.fit(X=np.array([[1, 1], [2, 2]]))
    assert len(outlier_remover.transform_train(np.array([[1, 1], [2, 2]]))) == 2


def test_remove_outlier(mocker):
    mock_outlier_detector = mocker.Mock()
    mock_outlier_detector.fit.return_value = None
    mock_outlier_detector.predict.return_value = np.array([-1])
    mocker.patch("sklego.meta.outlier_remover.clone").return_value = mock_outlier_detector

    outlier_remover = OutlierRemover(outlier_detector=mock_outlier_detector, refit=True)
    outlier_remover.fit(X=np.array([[5, 5]]))
    assert len(outlier_remover.transform_train(np.array([[0, 0]]))) == 0


def test_do_not_refit(mocker):
    mock_outlier_detector = mocker.Mock()
    mock_outlier_detector.fit.return_value = None
    mock_outlier_detector.predict.return_value = np.array([-1])
    mocker.patch("sklego.meta.outlier_remover.clone").return_value = mock_outlier_detector

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
