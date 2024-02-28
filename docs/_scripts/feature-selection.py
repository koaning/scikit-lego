from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")


_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

# --8<-- [start:mrmr-commonimports]
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklego.feature_selection import MaximumRelevanceMinimumRedundancy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# sns.set_theme(style='darkgrid')
# --8<-- [end:mrmr-commonimports]

# --8<-- [start:mrmr-intro]

# Download MNIST dataset using scikit-learn
mnist = fetch_openml("mnist_784", cache=True)

# Assign features and labels
X_pd, y_pd = mnist["data"], mnist["target"].astype(int)

X, y = X_pd.to_numpy(), y_pd.to_numpy()
t_t_s_params = {'test_size': 10000, 'random_state': 42}
X_train, X_test, y_train, y_test = train_test_split(X, y, **t_t_s_params)
X_train = X_train.reshape(60000, 28 * 28)
X_test = X_test.reshape(10000, 28 * 28)

def smile_relevance(X, y):
    rows = 28
    cols = 28
    smiling_face = np.zeros((rows, cols), dtype=int)

    # Set the values for the eyes, nose,
    # and mouth with adjusted positions and sizes
    # Left eye
    smiling_face[10:13, 8:10] = 1
    # Right eye
    smiling_face[10:13, 18:20] = 1
    # Upper part of the mouth
    smiling_face[18:20, 10:18] = 1
    # Left edge of the open mouth
    smiling_face[16:18, 8:10] = 1
    # Right edge of the open mouth
    smiling_face[16:18, 18:20] = 1

    # Add the nose as four pixels one pixel higher
    smiling_face[14, 13:15] = 1
    smiling_face[27, :] = 1
    return smiling_face.reshape(rows * cols,)


def smile_redundancy(X, selected, left):
    return np.ones(len(left))


K = 38
mrmr = MaximumRelevanceMinimumRedundancy(k=K,
                                         kind="auto",
                                         redundancy_func="p",
                                         relevance_func="f")
mrmr_s = MaximumRelevanceMinimumRedundancy(k=K,
                                           redundancy_func=smile_redundancy,
                                           relevance_func=smile_relevance)

f = f_classif(X_train ,y_train.reshape(60000,))[0]
f_features = np.argsort(np.nan_to_num(f, nan=np.finfo(float).eps))[-K:]
mi = mutual_info_classif(X_train, y_train.reshape(60000,))
mi_features = np.argsort(np.nan_to_num(mi, nan=np.finfo(float).eps))[-K:]
mrmr_features = mrmr.fit(X_train, y_train).selected_features_
mrmr_smile_features = mrmr_s.fit(X_train, y_train).selected_features_

# --8<-- [end:mrmr-intro]
# --8<-- [start:mrmr-selected-features]
# Define features dictionary
features = {
    "f_classif": f_features,
    "mutual_info": mi_features,
    "mrmr": mrmr_features,
    "mrmr_smile": mrmr_smile_features,
}
for name, s_f in features.items():
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train[:, s_f], y_train.squeeze())
    y_pred = model.predict(X_test[:, s_f])
    print(f"Feature selection method: {name}")
    print(round(f1_score(y_test, y_pred, average="weighted"), 3))

# --8<-- [end:mrmr-selected-features]

# --8<-- [start:mrmr-plots]
# Create figure and axes for the plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Iterate through the features dictionary and plot the images
for idx, (name, s_f) in enumerate(features.items()):
    row = idx // 2
    col = idx % 2

    a = np.zeros(28 * 28)
    a[s_f] = 1
    ax = axes[row, col]
    plot_= sns.heatmap(a.reshape(28, 28), cmap="binary", ax=ax, cbar=False)
    ax.set_title(name)




# --8<-- [end:mrmr-plots]
plt.tight_layout()
plt.savefig(_static_path / "mrmr-feature-selection-mnist.png")
plt.clf()
