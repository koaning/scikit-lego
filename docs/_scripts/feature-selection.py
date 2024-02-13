from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklego.feature_selection.mrmr import MaximumRelevanceMinimumRedundancy

# --8<-- [start:mrmr]

# Download MNIST dataset using scikit-learn
mnist = fetch_openml("mnist_784", cache=True)

# Assign features and labels
X_pd, y_pd = mnist["data"], mnist["target"]

X, y = X_pd.to_numpy(), y_pd.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
X_train = X_train.reshape(60000, 28 * 28)
X_test = X_test.reshape(10000, 28 * 28)


from scipy.spatial.distance import cosine


def _redundancy_cosine_scipy(X, selected, left):
    if len(selected) == 0:
        return np.ones(len(left))

    score = np.array([np.sum([cosine(X[:, _s], X[:, _l]) for _s in selected]) for _l in left])
    score[np.isclose(score, 0.0, atol=np.finfo(float).eps, rtol=0)] = np.finfo(float).eps
    return np.array(score)


def smile_relevance(X, y):
    rows = 28
    cols = 28
    smiling_face = np.zeros((rows, cols), dtype=int)

    # Set the values for the eyes, nose, and mouth with adjusted positions and sizes
    smiling_face[10:13, 8:10] = 1  # Left eye
    smiling_face[10:13, 18:20] = 1  # Right eye
    smiling_face[16:18, 10:18] = 1  # Upper part of the mouth
    smiling_face[18:20, 8:10] = 1  # Left edge of the open mouth
    smiling_face[18:20, 18:20] = 1  # Right edge of the open mouth

    # Add the nose as four pixels one pixel higher
    smiling_face[14, 13:15] = 1
    smiling_face[27, :] = 1
    return smiling_face.reshape(
        rows * cols,
    )


def smile_redundancy(X, selected, left):
    return np.ones(len(left))


K = 35
mrmr = MaximumRelevanceMinimumRedundancy(k=K, kind="auto", redundancy_func="p", relevance_func="f")
mrmr_cosine = MaximumRelevanceMinimumRedundancy(
    k=K, kind="auto", redundancy_func=_redundancy_cosine_scipy, relevance_func="f"
)
mrmr_smile = MaximumRelevanceMinimumRedundancy(k=K, redundancy_func=smile_redundancy, relevance_func=smile_relevance)

f = f_classif(
    X_train,
    y_train.reshape(
        60000,
    ),
)[0]
f_features = np.argsort(np.nan_to_num(f, nan=np.finfo(float).eps))[-K:]
mi = mutual_info_classif(
    X_train,
    y_train.reshape(
        60000,
    ),
)
mi_features = np.argsort(np.nan_to_num(mi, nan=np.finfo(float).eps))[-K:]
mrmr_features = mrmr.fit(X_train, y_train).selected_features_
mrmr_cos_features = mrmr_cosine.fit(X_train, y_train).selected_features_
mrmr_smile_features = mrmr_smile.fit(X_train, y_train).selected_features_


features = {
    "f_classif": f_features,
    "mutual_info": mi_features,
    "mrmr": mrmr_features,
    "mrmr_cosine": mrmr_cos_features,
    "mrmr_smile": mrmr_smile_features,
}
for name, s_f in features.items():
    model = HistGradientBoostingClassifier()
    model.fit(X_train[:, s_f], y_train.squeeze())
    y_pred = model.predict(X_test[:, s_f])
    print(f1_score(y_test, y_pred, average="weighted"))

import matplotlib.pyplot as plt
import numpy as np

# Create figure and axes for the plots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Define features dictionary
features = {
    "f_classif": f_features,
    "mutual_info": mi_features,
    "mrmr": mrmr_features,
    "mrmr_cos": mrmr_cos_features,
    "mrmr_smile": mrmr_smile_features,
}

# Iterate through the features dictionary and plot the images
for idx, (name, s_f) in enumerate(features.items()):
    row = idx // 3  # Calculate the row index
    col = idx % 3  # Calculate the column index

    a = np.zeros(28 * 28)
    a[s_f] = 1
    ax = axes[row, col]
    ax.imshow(a.reshape(28, 28), cmap="binary", vmin=0, vmax=1)
    ax.set_title(name)

# --8<-- [end:mrmr]

plt.tight_layout()
plt.savefig(_static_path / "mrmr-feature-selection-mnist.png")
plt.clf()
