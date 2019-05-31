import pytest
from pandas.tests.extension.numpy_.test_numpy_nested import np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import check_X_y, estimator_checks

from sklego.common import flatten
from sklego.meta import OutlierRemover
from sklego.mixture import GMMOutlierDetector
from tests.conftest import general_checks, nonmeta_checks, transformer_checks 


@pytest.mark.parametrize("test_fn", flatten([
    estimator_checks.check_transformers_unfitted,
    estimator_checks.check_fit2d_predict1d,
    estimator_checks.check_fit2d_1sample,
    estimator_checks.check_fit2d_1feature,
    estimator_checks.check_fit1d,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_dont_overwrite_parameters,
    estimator_checks.check_transformers_unfitted
]))
def test_estimator_checks(test_fn):
    outlier_remover = OutlierRemover(outlier_detector=GMMOutlierDetector(), refit=True)
    test_fn(OutlierRemover.__name__, outlier_remover)


@pytest.fixture
def mock_outlier_detector(mocker):
    return mocker.Mock()


def test_no_outliers(mock_outlier_detector, mocker):
    mock_outlier_detector.fit.return_value = None
    mock_outlier_detector.predict.return_value = np.array([[1,1]])
    mocker.patch('sklego.meta.clone').return_value = mock_outlier_detector
    
    outlier_remover = OutlierRemover(outlier_detector=mock_outlier_detector, refit=True)
    outlier_remover.fit(X=np.array([[1,1], [2,2]]))
    assert len(outlier_remover.transform_train(np.array([[1,1], [2,2]]))) == 2
    

def test_remove_outlier(mock_outlier_detector, mocker):
    mock_outlier_detector.fit.return_value = None
    mock_outlier_detector.predict.return_value = np.array([[-1]])
    mocker.patch('sklego.meta.clone').return_value = mock_outlier_detector
    
    outlier_remover = OutlierRemover(outlier_detector=mock_outlier_detector, refit=True)
    outlier_remover.fit(X=np.array([[5,5]]))
    assert len(outlier_remover.transform_train(np.array([[0, 0]]))) == 0
