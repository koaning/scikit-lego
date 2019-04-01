from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import confusion_matrix

import numpy as np

class EnsembleModel():
    """
    A classifier that takes the uncertainty p(class | model) of a classifier into account.
    Based on the requirements of issue #23 of koaning/scikit-lego
    """

    def __init__(self, estimator=None, alpha=0.5, X_test=None, y_test=None):
        """

        :param estimator: sklean classifier; (eg. LogisticRegression, Decision Tree etc.)
        :param alpha: hyperparameter; needs to be optimized to increase classification accuracy
        :param X_test: array of feature values that will be used to test the classifier and create a confusion matrix
        :param y_test: array of target values that will be used to test the classifier and create a confusion matrix
        """
        self.estimator = estimator
        self.alpha = alpha
        self.X_test = X_test
        self.y_test = y_test

    def fit(self, X, y):
        """

       :param X: array of feature values used for training the classifier
       :param y: array of target values used for training the classifier
       :return:
                - trained classifier (self.orig_estimator)
                - confusion matrix populated with the probabilities of each real value vs model prediction value
                  of the target variable (self.confusion_matrix)
                - updated prediction probabilities of each target value made by the trained classifier
                  (self.prob_matrix)
       """
        self.estimator_ = clone(self.estimator)
        self.orig_estimator_ = self.estimator_.fit(X, y)

        y_pred = self.orig_predict_(self.X_test)
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

        cfm = self.confusion_matrix.T
        self.prob_matrix = (cfm.T/cfm.sum(axis=1)).T

        return self

    def orig_predict_(self, X):
        """

        :param X: array of feature values upon which a prediction is to be made
        :return: prediction of the target value made by the original trained classifier
        """
        return self.orig_estimator_.predict(X)

    def orig_predict_proba_(self, X):
        """

        :param X: array of feature values upon which a prediction is to be made
        :return: prediction probabilities of each of the target value made by the original trained classifier
        """
        return self.orig_estimator_.predict_proba(X)

    def p(self, X):
        """

        :param X: array of feature values upon which a prediction is to be made
        :return: original prediction probabilities of each target value made by the trained classifier
        """
        p = self.orig_predict_proba_(X)
        return p

    def p_star(self, X):
        """

        :param X: array of feature values upon which a prediction is to be made
        :return: updated prediction probabilities of each target value made by the trained classifier
        """
        p = self.p(X)
        return  p @ self.prob_matrix

    def predict_proba(self, X):
        """

        :param X: array of feature values upon which a prediction is to be made
        :return: prediction probabilities of each of the target value made by the updated trained classifier;
                 based on the original prediction, updated prediction, and the (recommendably tuned) alpha parameter
        """
        check_is_fitted(self, 'estimator_')
        p = self.p(X)
        p_star = self.p_star(X)
        return (1-self.alpha)*p + self.alpha*p_star

    def predict(self, X):
        """

        :param X: array of feature values upon which a prediction is to be made
        :return: prediction of the target value made by the updated trained classifier;
                 based on the original prediction, updated prediction, and the (recommendably tuned) alpha parameter
        """
        predict_proba = self.predict_proba(X)
        prediction = np.argmax(predict_proba, axis=1)
        return prediction