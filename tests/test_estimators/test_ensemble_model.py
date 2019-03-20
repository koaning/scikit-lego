from sklego.ensemble_model import EnsembleModel
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Data
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = EnsembleModel(X_test = X_test, y_test = y_test)
model.fit(X_train, y_train)

def test_alpha():
    assert model.alpha >= 0
    assert model.alpha <= 1


def test_predict_proba():
    assert model.predict([X_test[0]]).shape[0] == 1


def test_confusion_matrix():
    assert model.confusion_matrix().shape[0] == model.confusion_matrix().shape[1]


def test_make_prediction():
    assert model.make_prediction([X_test[0]]).shape[0] == 1




