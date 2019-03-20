from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class EnsembleModel():
    """
    A classifier that takes the uncertainty p(class | model) of a classifier into account.
    Based on the requirements of issue #23 of koaning/scikit-lego
    """

    def __init__(self, X_test, y_test, estimator=LogisticRegression(), alpha=0.5):
        """

        :param X_test: array of feature values that will be used to test the classifier and create a confusion matrix
        :param y_test: array of target values that will be used to test the classifier and create a confusion matrix
        :param estimator: sklean classifier;
                          default one is LogisticRegression, but others can be used too (eg. Decision Tree)
        :param alpha: hyperparameter; needs to be optimized to increase classification accuracy
        """
        self.estimator = estimator
        self.alpha = alpha
        self.X_test = X_test
        self.y_test = y_test

    def fit(self, X, y):
        """

        :param X: array of feature values used for training the classifier
        :param y: array of target values used for training the classifier
        :return: trained classifier (estimator)
        """
        self.fitted_estimator_ = self.estimator.fit(X, y)

        return self

    def predict(self, X):
        """

        :param X: array of feature values upon which a prediction is to be made
        :return: prediction of the target value made by the trained classfifier
        """
        return self.fitted_estimator_.predict(X)

    def predict_proba(self, X):
        """

        :param X: array of feature values upon which a prediction is to be made
        :return: prediction probabilities of each of the target value made by the trained classifier
        """
        return self.fitted_estimator_.predict_proba(X)

    def confusion_matrix(self):
        """

        :return: Confusion matrix of the trained classifier based on test data
        """
        y_pred = self.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)

    def prob_matrix(self):
        """

        :return: Confusion matrix populated with the probabilities of each real value vs model prediction value
                 of the target variable
        """
        cfm = self.confusion_matrix().T
        return (cfm.T / cfm.sum(axis=1)).T

    def p_star(self, X):
        """

        :param X: array of feature values upon which a prediction is to be made
        :return: Updated prediction probabilities of each of the target value made by the trained classifier
        """
        p = self.predict_proba(X)
        prob_matrix = self.prob_matrix()
        return p @ prob_matrix

    def make_prediction(self, X):
        """

        :param X: array of feature values upon which a prediction is to be made
        :return: Final prediction; based on the initial prediction, modified prediction,
                 and the (recommendably tuned) alpha parameter
        """
        p = self.predict_proba(X)
        p_star = self.p_star(X)
        return (1 - self.alpha) * p + self.alpha * p_star
