# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.19.10",
#     "numpy==2.4.2",
#     "scikit-learn==1.8.0",
#     "scipy==1.17.0",
# ]
# ///
import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Isolation Forest Encoder

        This notebook demonstrates the `IsolationForestEncoder`, which generates
        embeddings from tabular data using Isolation Forest tree paths.

        The encoder tracks the path each sample takes through the trees in an
        Isolation Forest, encoding left/right decisions as 0/1 in a sparse binary array.
        """
    )
    return (mo,)


@app.cell
def _():
    ## Export
    import numpy as np
    from scipy import sparse
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.ensemble import IsolationForest
    from sklearn.utils.validation import check_is_fitted

    return BaseEstimator, IsolationForest, TransformerMixin, check_is_fitted, np, sparse


@app.cell
def _(BaseEstimator, IsolationForest, TransformerMixin, check_is_fitted, np, sparse):
    ## Export
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

    return (IsolationForestEncoder,)


@app.cell
def _(IsolationForestEncoder, mo, np):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Create some tabular data
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Create a pipeline with the forest encoder
    pipe = make_pipeline(
        StandardScaler(),
        IsolationForestEncoder(n_estimators=50),
        LogisticRegression(),
    )

    # Fit and predict
    pipe.fit(X, y)
    predictions = pipe.predict(X)
    accuracy = np.mean(predictions == y)

    mo.md(
        f"""
        ## Quick Demo

        Trained a pipeline with `StandardScaler -> IsolationForestEncoder(n_estimators=50) -> LogisticRegression`
        on 200 samples with 5 features.

        **Training accuracy**: {accuracy:.2%}
        """
    )
    return LogisticRegression, StandardScaler, accuracy, make_pipeline, pipe, predictions, rng, X, y


@app.cell
def _(IsolationForestEncoder, X, mo, np):
    # Show the shape of the encoding
    enc = IsolationForestEncoder(n_estimators=50, random_state=42)
    enc.fit(X)
    X_encoded = enc.transform(X)

    mo.md(
        f"""
        ## Encoding Details

        - Input shape: `{X.shape}`
        - Encoded shape: `{X_encoded.shape}`
        - Encoding density: `{X_encoded.nnz / np.prod(X_encoded.shape):.2%}`

        The encoder creates a sparse binary matrix where each column represents
        a left/right decision at a node in one of the isolation trees.
        """
    )
    return X_encoded, enc


if __name__ == "__main__":
    app.run()
