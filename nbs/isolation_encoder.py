# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "scikit-learn==1.8.0",
#     "scipy==1.17.0",
#     "anywidget",
#     "traitlets",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="columns")


@app.cell(column=0)
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return


@app.cell
def _(IsolationForestEncoder):
    from sklearn.datasets import make_moons

    X, _ = make_moons(n_samples=10000, noise=0.15, random_state=42)

    enc = IsolationForestEncoder(n_estimators=2, max_features=2, max_samples=100)
    out = enc.fit_transform(X)
    return X, enc, out


@app.cell
def _(mo, out):
    slider = mo.ui.slider(0, out.shape[1] - 1, 1)
    slider
    return (slider,)


@app.cell
def _(X, out, plt, slider):
    selected = X[(out[:, slider.value] == 1).toarray()[:, 0], :]

    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(selected[:, 0], selected[:, 1], c="orange")
    return


@app.cell
def _(TreeWidget, X, enc, mo):
    tree_widget = mo.ui.anywidget(TreeWidget(enc.forest_.estimators_[0], X))
    tree_widget
    return (tree_widget,)


@app.cell
def _():
    import matplotlib.pylab as plt

    return (plt,)


@app.cell(hide_code=True)
def _(X, enc, plt, tree_widget):
    node_id = tree_widget.value["selected_node"]
    estimator = enc.forest_.estimators_[0]

    if node_id >= 0:
        mask_arr = estimator.decision_path(X)[:, node_id].toarray().ravel().astype(bool)
    else:
        mask_arr = None

    plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
    plt.scatter(X[mask_arr, 0], X[mask_arr, 1], c="orange")
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    import pathlib
    import anywidget
    import traitlets

    class TreeWidget(anywidget.AnyWidget):
        _esm = pathlib.Path(__file__).parent / "ensemble" / "widget.js"
        _css = pathlib.Path(__file__).parent / "ensemble" / "widget.css"

        tree_data = traitlets.Dict({}).tag(sync=True)
        selected_node = traitlets.Int(-1).tag(sync=True)

        def __init__(self, estimator, X, feature_names=None, **kwargs):
            self._estimator = estimator
            self._X = X
            tree_data = self._compute_layout(estimator.tree_, feature_names)
            super().__init__(tree_data=tree_data, **kwargs)

        def _compute_layout(self, tree, feature_names=None):
            import numpy as np

            children_left = tree.children_left
            children_right = tree.children_right
            feature = tree.feature
            threshold = tree.threshold

            # Compute actual sample counts from full X, not just training subset
            decision_paths = self._estimator.decision_path(self._X)
            actual_samples = np.array(decision_paths.sum(axis=0)).ravel()

            depths = {}
            inorder_pos = {}
            counter = [0]

            def traverse(node_id, depth):
                depths[node_id] = depth
                if children_left[node_id] != -1:
                    traverse(children_left[node_id], depth + 1)
                inorder_pos[node_id] = counter[0]
                counter[0] += 1
                if children_right[node_id] != -1:
                    traverse(children_right[node_id], depth + 1)

            traverse(0, 0)

            max_depth = max(depths.values())
            max_pos = max(inorder_pos.values())
            padding = 30
            width = 700
            height = max(max_depth * 50 + 2 * padding, 200)

            nodes = []
            edges = []
            for i in range(tree.node_count):
                x = padding + (inorder_pos[i] / max(max_pos, 1)) * (width - 2 * padding)
                y = padding + (depths[i] / max(max_depth, 1)) * (height - 2 * padding)
                is_leaf = children_left[i] == -1
                n_samples = int(actual_samples[i])

                if is_leaf:
                    label = f"n={n_samples}"
                else:
                    feat_name = f"x[{feature[i]}]" if feature_names is None else feature_names[feature[i]]
                    label = f"{feat_name} <= {threshold[i]:.2f}"

                nodes.append({
                    "id": i, "x": float(x), "y": float(y),
                    "label": label, "samples": n_samples, "is_leaf": is_leaf,
                })
                if not is_leaf:
                    edges.append({"source": i, "target": int(children_left[i])})
                    edges.append({"source": i, "target": int(children_right[i])})

            return {"nodes": nodes, "edges": edges, "width": int(width), "height": int(height)}

        def compute_mask(self, node_id):
            """Return boolean array: which samples from X reach this node."""
            import numpy as np
            if node_id < 0:
                return np.zeros(self._X.shape[0], dtype=bool)
            decision_paths = self._estimator.decision_path(self._X)
            return decision_paths[:, node_id].toarray().ravel().astype(bool)

    return (TreeWidget,)


@app.cell(column=2)
def _(
    BaseEstimator,
    IsolationForest,
    TransformerMixin,
    check_is_fitted,
    np,
    sparse,
):
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
def _():
    ## Export
    import numpy as np
    from scipy import sparse
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.ensemble import IsolationForest
    from sklearn.utils.validation import check_is_fitted

    return (
        BaseEstimator,
        IsolationForest,
        TransformerMixin,
        check_is_fitted,
        np,
        sparse,
    )


if __name__ == "__main__":
    app.run()
