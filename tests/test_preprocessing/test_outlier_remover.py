import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.mixture import GMMOutlierDetector
from sklego.preprocessing import OutlierRemover


@parametrize_with_checks(
    [
        OutlierRemover(outlier_detector=GMMOutlierDetector(), refit=True),
        OutlierRemover(outlier_detector=IsolationForest(), refit=True),
    ]
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
