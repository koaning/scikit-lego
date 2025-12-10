# Quickstart — scikit-lego in 5 minutes

This quickstart demonstrates a small end-to-end machine learning workflow  
(dataset → preprocessing → pipeline → model → evaluation).  
It is intentionally minimal — follow the links below for detailed examples and explanations.

---

## Requirements

```bash
pip install scikit-learn scikit-lego
```

## 1. Load data

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

## 2. Preprocessing (sklearn + scikit-lego)

You can mix standard scikit-learn transformers with scikit-lego transformers.
Here we use `ColumnSelector` from scikit-lego to select specific columns before applying sklearn preprocessing.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklego.preprocessing import ColumnSelector

# Use ColumnSelector to select specific columns (scikit-lego feature)
preprocessor = Pipeline([
    ("select", ColumnSelector(["sepal length (cm)", "petal length (cm)"])),  # scikit-lego
    ("impute", SimpleImputer()),  # sklearn
    ("scale", StandardScaler()),  # sklearn
])
```

## 3. Build pipeline & train model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
```

## 4. Evaluate the model

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

## 5. Where scikit-lego adds value

scikit-lego provides:

- Shortcuts and helpers for column selection, encoding, and feature engineering
- Utility transformers that plug cleanly into sklearn pipelines
- Additional models, preprocessing tools, and diagnostics

Explore the User Guide sections below for full examples.

## 6. Learn more — User Guide references

- **Preprocessing**: [user-guide/preprocessing.md](preprocessing.md)
- **Pipelines & Utilities**: [user-guide/pandas-pipelines.md](pandas-pipelines.md)
- **Meta Models**: [user-guide/meta-models.md](meta-models.md)

**Tip**: Keep `random_state=0` for deterministic results.
If you'd like, replace the sklearn-only pipeline above with a scikit-lego-specific transformer (e.g., ColumnSelector).
