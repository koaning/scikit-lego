from sklego.ensemble_model import EnsembleModel
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Data
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = EnsembleModel(estimator=LogisticRegression(), X_test = X_test, y_test = y_test)
model.fit(X_train, y_train)


def test_alpha():
   if model.alpha > 1:
    assert False


def test_confusion_matrix():
    assert model.confusion_matrix.shape[0] == model.confusion_matrix.shape[1]


def test_predict_proba():
    assert model.predict_proba([X_test[0]]).shape[0] == 1


def test_predict():
    assert model.predict([X_test[0]]).shape[0] == 1




