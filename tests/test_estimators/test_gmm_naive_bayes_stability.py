import numpy as np
import pytest

from sklego.naive_bayes import BayesianGaussianMixtureNB, GaussianMixtureNB


@pytest.mark.parametrize("estimator", [GaussianMixtureNB, BayesianGaussianMixtureNB])
@pytest.mark.parametrize("labels", [[2, 7], ["left", "right"]])
def test_tail_probabilities_remain_finite_and_predict_the_nearer_class(estimator, labels):
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([labels[0], labels[0], labels[1], labels[1]])
    model = estimator(random_state=0).fit(X, y)
    X_test = np.array([[-1000.0], [1000.0]])

    probabilities = model.predict_proba(X_test)

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(probabilities, np.eye(2))
    np.testing.assert_array_equal(model.predict(X_test), labels)


@pytest.mark.parametrize("estimator", [GaussianMixtureNB, BayesianGaussianMixtureNB])
def test_high_density_features_do_not_overflow(estimator):
    values = np.array([-1e-8, 0.0, 1e-8, -1e-8, 0.0, 1e-8])
    X = np.tile(values[:, None], (1, 64))
    model = estimator(random_state=0, reg_covar=1e-12).fit(X, [0, 0, 0, 1, 1, 1])

    probabilities = model.predict_proba(np.zeros((1, 64)))

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities, [[0.5, 0.5]])


@pytest.mark.parametrize("estimator", [GaussianMixtureNB, BayesianGaussianMixtureNB])
def test_regular_probabilities_match_direct_likelihood_normalization(estimator):
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    model = estimator(random_state=0).fit(X, [0, 0, 1, 1])
    X_test = np.array([[0.0, 0.0], [-0.5, -0.5], [0.5, 0.5]])
    likelihoods = np.column_stack(
        [
            np.prod(
                [np.exp(component.score_samples(X_test[:, [i]])) for i, component in enumerate(components)],
                axis=0,
            )
            for components in model.gmms_.values()
        ]
    )

    np.testing.assert_allclose(model.predict_proba(X_test), likelihoods / likelihoods.sum(axis=1, keepdims=True))
