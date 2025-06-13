import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.preprocessing import LinearEmbedder


@parametrize_with_checks([LinearEmbedder()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_estimator_configuration():
    """Test default and custom estimator configuration."""
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    
    # Test default estimator
    embedder_default = LinearEmbedder()
    embedder_default.fit(X, y)
    
    assert isinstance(embedder_default.estimator_, Ridge)
    assert not embedder_default.estimator_.fit_intercept
    assert embedder_default.n_features_in_ == 5
    
    # Test custom estimator
    custom_ridge = Ridge(alpha=10.0, fit_intercept=True)
    embedder_custom = LinearEmbedder(estimator=custom_ridge)
    embedder_custom.fit(X, y)
    
    assert embedder_custom.estimator_ is custom_ridge
    assert embedder_custom.estimator_.alpha == 10.0
    assert embedder_custom.estimator_.fit_intercept
    assert embedder_custom.n_features_in_ == 5


def test_basic_embedding_functionality():
    """Test embedding with regression, classification, and multi-output data."""
    # Test regression embedding
    X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
    embedder_reg = LinearEmbedder()
    
    X_embedded_reg = embedder_reg.fit_transform(X_reg, y_reg)
    assert X_embedded_reg.shape == X_reg.shape
    assert hasattr(embedder_reg, 'coef_')
    assert embedder_reg.coef_.shape == (1, X_reg.shape[1])
    
    # Check transform works separately
    X_embedded_reg_2 = embedder_reg.transform(X_reg)
    np.testing.assert_array_equal(X_embedded_reg, X_embedded_reg_2)
    
    # Test classification embedding
    X_clf, y_clf = make_classification(n_samples=100, n_features=10, random_state=42)
    embedder_clf = LinearEmbedder(estimator=LogisticRegression(fit_intercept=False, max_iter=1000))
    
    X_embedded_clf = embedder_clf.fit_transform(X_clf, y_clf)
    assert X_embedded_clf.shape == X_clf.shape
    assert hasattr(embedder_clf, 'coef_')
    assert embedder_clf.coef_.shape == (1, X_clf.shape[1])
    
    # Test multi-output regression
    X_multi, y_multi = make_regression(n_samples=100, n_features=5, n_targets=3, random_state=42)
    embedder_multi = LinearEmbedder()
    
    X_embedded_multi = embedder_multi.fit_transform(X_multi, y_multi)
    assert X_embedded_multi.shape == X_multi.shape
    assert hasattr(embedder_multi, 'coef_')
    # Check that coefficients are averaged across outputs
    assert embedder_multi.coef_.shape == (1, X_multi.shape[1])


def test_input_validation_errors():
    """Test error conditions for input validation."""
    X_train, y_train = make_regression(n_samples=50, n_features=5, random_state=42)
    
    # Test that transform fails when called before fit
    embedder_unfitted = LinearEmbedder()
    with pytest.raises(Exception):  # Should raise NotFittedError or similar
        embedder_unfitted.transform(X_train)
    
    # Test that transform checks for consistent number of features
    embedder_fitted = LinearEmbedder()
    embedder_fitted.fit(X_train, y_train)
    
    X_wrong_features = np.random.randn(10, 3)  # Different number of features
    with pytest.raises(ValueError):
        embedder_fitted.transform(X_wrong_features)


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


def test_single_feature():
    """Test embedding with single feature edge case."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    
    embedder = LinearEmbedder()
    X_embedded = embedder.fit_transform(X, y)
    
    assert X_embedded.shape == (4, 1)
    assert hasattr(embedder, 'coef_')
    assert embedder.coef_.shape == (1, 1)