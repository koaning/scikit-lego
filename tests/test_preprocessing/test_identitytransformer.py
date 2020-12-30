import numpy as np
import pytest

from sklego.common import flatten
from sklego.preprocessing import IdentityTransformer

from tests.conftest import select_tests, transformer_checks, general_checks, nonmeta_checks


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, transformer_checks]),
        exclude=[
            "check_sample_weights_invariance",
        ]
    )
)
def test_estimator_checks(test_fn):
    test_fn(IdentityTransformer.__name__, IdentityTransformer())


def test_same_values(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    X_new = IdentityTransformer().fit_transform(X)
    assert np.isclose(X, X_new).all()
