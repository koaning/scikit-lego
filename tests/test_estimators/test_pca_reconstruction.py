import pytest

from sklego.common import flatten
from sklego.decomposition import PCAOutlierDetection
from tests.conftest import nonmeta_checks, general_checks, outlier_checks


@pytest.mark.parametrize(
    "test_fn",
    flatten(
        [
            nonmeta_checks,
            general_checks,
            outlier_checks
        ]
    ),
)
def test_estimator_checks(test_fn):
    outlier_mod = PCAOutlierDetection(n_components=2, threshold=0.1)
    test_fn(PCAOutlierDetection.__name__, outlier_mod)
