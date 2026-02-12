
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted

class IsolationForestEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder that generates embeddings from tabular data using Isolation Forest.

    This encoder tracks the path each sample takes through the trees in an
    Isolation Forest, encoding left/right decisions as 0/1 in a sparse binary array.
    Each tree contributes a sequence of binary decisions that form part of the
    final embedding.

    Arguments:
        n_estimators: The number of base estimators in the ensemble
        max_samples: The number of samples to draw to train each base estimator
        max_features: The number of features to draw to train each base estimator
        contamination: The proportion of outliers in the dataset
        random_state: Random state for reproducibility (default: 42)
        n_jobs: Number of parallel jobs to run

    **Usage**:

    ```python
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # Create some tabular data
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Create a pipeline with the forest encoder
    pipe = make_pipeline(
        StandardScaler(),
        IsolationForestEncoder(n_estimators=50),
        LogisticRegression()
    )

    # Fit and predict
    pipe.fit(X, y)
    predictions = pipe.predict(X)
    ```
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        max_features=1.0,
        contamination="auto",
        random_state=42,
        n_jobs=None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the Isolation Forest on the training data."""
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.forest_.fit(X)
        return self

    def transform(self, X, y=None):
        """
        Transform the input data into tree path embeddings.

        Returns a sparse binary matrix where each row represents a sample
        and columns represent binary decisions (0=left, 1=right) at each
        node across all trees.
        """
        check_is_fitted(self, "forest_")

        # Get the number of samples
        n_samples = X.shape[0]

        # List to store the path encoding for each tree
        all_paths = []

        # For each tree in the forest
        for estimator in self.forest_.estimators_:
            # Get the decision path for all samples
            # This returns a sparse matrix where each row is a sample
            # and columns indicate which nodes were visited
            decision_path = estimator.decision_path(X)

            # Get the tree structure
            tree = estimator.tree_

            # For each sample, encode the path as left/right decisions
            paths_for_tree = []
            for i in range(n_samples):
                # Get the nodes visited by this sample
                node_indicator = decision_path[i].toarray().ravel()
                node_ids = np.where(node_indicator)[0]

                # Encode the path (excluding the root and leaf)
                path_encoding = []
                for j in range(len(node_ids) - 1):
                    current_node = node_ids[j]
                    next_node = node_ids[j + 1]

                    # Check if we went left (0) or right (1)
                    if tree.children_left[current_node] == next_node:
                        path_encoding.append(0)
                    else:
                        path_encoding.append(1)

                paths_for_tree.append(path_encoding)

            # Pad paths to have the same length within this tree
            max_depth = max(len(p) for p in paths_for_tree)
            padded_paths = []
            for path in paths_for_tree:
                # Pad with -1 (which we'll handle specially)
                padded = path + [-1] * (max_depth - len(path))
                padded_paths.append(padded)

            all_paths.append(np.array(padded_paths))

        # Concatenate paths from all trees
        # Each sample now has a sequence of decisions from all trees
        full_encoding = np.hstack(all_paths)

        # Replace -1 (padding) with 0 for simplicity
        full_encoding[full_encoding == -1] = 0

        # Convert to sparse matrix for efficiency
        sparse_encoding = sparse.csr_matrix(full_encoding)

        return sparse_encoding
