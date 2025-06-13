import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.preprocessing import LinearEmbedder


@parametrize_with_checks([LinearEmbedder()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_default_estimator():
    """Test that default estimator is Ridge with fit_intercept=False."""
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    embedder = LinearEmbedder()
    embedder.fit(X, y)
    
    assert isinstance(embedder.estimator_, Ridge)
    assert not embedder.estimator_.fit_intercept


def test_custom_estimator():
    """Test using a custom estimator."""
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    custom_ridge = Ridge(alpha=10.0, fit_intercept=True)
    embedder = LinearEmbedder(estimator=custom_ridge)
    embedder.fit(X, y)
    
    assert embedder.estimator_ is custom_ridge
    assert embedder.estimator_.alpha == 10.0
    assert embedder.estimator_.fit_intercept


def test_regression_embedding():
    """Test embedding with regression data."""
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    embedder = LinearEmbedder()
    
    # Fit and transform
    X_embedded = embedder.fit_transform(X, y)
    
    # Check shape is preserved
    assert X_embedded.shape == X.shape
    
    # Check that coefficients are stored
    assert hasattr(embedder, 'coef_')
    assert embedder.coef_.shape == (1, X.shape[1])
    
    # Check transform works separately
    X_embedded_2 = embedder.transform(X)
    np.testing.assert_array_equal(X_embedded, X_embedded_2)


def test_classification_embedding():
    """Test embedding with classification data."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    # Use logistic regression for classification
    embedder = LinearEmbedder(estimator=LogisticRegression(fit_intercept=False, max_iter=1000))
    
    # Fit and transform
    X_embedded = embedder.fit_transform(X, y)
    
    # Check shape is preserved
    assert X_embedded.shape == X.shape
    
    # Check that coefficients are stored
    assert hasattr(embedder, 'coef_')
    assert embedder.coef_.shape == (1, X.shape[1])


def test_multioutput_regression():
    """Test embedding with multi-output regression."""
    X, y = make_regression(n_samples=100, n_features=5, n_targets=3, random_state=42)
    embedder = LinearEmbedder()
    
    # Fit and transform
    X_embedded = embedder.fit_transform(X, y)
    
    # Check shape is preserved
    assert X_embedded.shape == X.shape
    
    # Check that coefficients are averaged across outputs
    assert hasattr(embedder, 'coef_')
    assert embedder.coef_.shape == (1, X.shape[1])


def test_coefficient_scaling():
    """Test that the embedding actually scales features by coefficients."""
    # Create simple data where we know the expected result
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    
    # Use a Ridge with alpha=0 to get OLS solution
    embedder = LinearEmbedder(estimator=Ridge(alpha=1e-10, fit_intercept=False))
    embedder.fit(X, y)
    
    # Manual calculation: X_embedded should be X * coef_
    X_embedded = embedder.transform(X)
    expected = X * embedder.coef_
    
    np.testing.assert_allclose(X_embedded, expected)


def test_fit_before_transform():
    """Test that transform fails when called before fit."""
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    embedder = LinearEmbedder()
    
    with pytest.raises(Exception):  # Should raise NotFittedError or similar
        embedder.transform(X)


def test_consistent_n_features():
    """Test that transform checks for consistent number of features."""
    X_train, y_train = make_regression(n_samples=50, n_features=5, random_state=42)
    X_test = np.random.randn(10, 3)  # Different number of features
    
    embedder = LinearEmbedder()
    embedder.fit(X_train, y_train)
    
    with pytest.raises(ValueError):
        embedder.transform(X_test)


def test_check_input_parameter():
    """Test the check_input parameter functionality."""
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    
    # With check_input=True (default)
    embedder_check = LinearEmbedder(check_input=True)
    embedder_check.fit(X, y)
    X_embedded_check = embedder_check.transform(X)
    
    # With check_input=False
    embedder_no_check = LinearEmbedder(check_input=False)
    embedder_no_check.fit(X, y)
    X_embedded_no_check = embedder_no_check.transform(X)
    
    # Results should be similar
    np.testing.assert_allclose(X_embedded_check, X_embedded_no_check)


def test_embedding_improves_representation():
    """Test that embedding can improve feature representation."""
    # Create a simple case where features have different scales
    np.random.seed(42)
    n_samples = 1000
    
    # Create features with different importance
    X1 = np.random.randn(n_samples, 1) * 0.1  # Low importance, small scale
    X2 = np.random.randn(n_samples, 1) * 10   # High importance, large scale
    X = np.hstack([X1, X2])
    
    # Target depends more on X1 (despite its small scale)
    y = 5 * X1.flatten() + 0.1 * X2.flatten() + 0.1 * np.random.randn(n_samples)
    
    embedder = LinearEmbedder()
    X_embedded = embedder.fit_transform(X, y)
    
    # The embedding should amplify the important feature (X1) relative to X2
    coef_ratio = abs(embedder.coef_[0, 0]) / abs(embedder.coef_[0, 1])
    original_scale_ratio = np.std(X[:, 0]) / np.std(X[:, 1])
    
    # The coefficient should partially correct for the scale difference
    assert coef_ratio > original_scale_ratio


def test_n_features_in_attribute():
    """Test that n_features_in_ is set correctly."""
    X, y = make_regression(n_samples=50, n_features=7, random_state=42)
    embedder = LinearEmbedder()
    embedder.fit(X, y)
    
    assert embedder.n_features_in_ == 7


def test_single_feature():
    """Test embedding with single feature."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    
    embedder = LinearEmbedder()
    X_embedded = embedder.fit_transform(X, y)
    
    assert X_embedded.shape == (4, 1)
    assert hasattr(embedder, 'coef_')
    assert embedder.coef_.shape == (1, 1)