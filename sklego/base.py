from sklearn.base import OutlierMixin


class ProbabilisticClassifierMeta(type):
    """Metaclass for `ProbabilisticClassifier`.

    This metaclass is responsible for checking whether a class can be considered a `ProbabilisticClassifier`.
    A class is considered a `ProbabilisticClassifier` if it has a "predict_proba" method.
    """

    def __instancecheck__(self, other):
        """Checks if the provided object is a `ProbabilisticClassifier`.

        Parameters
        ----------
        self : ProbabilisticClassifierMeta
            `ProbabilisticClassifierMeta` class.
        other : object
            The object to check for `ProbabilisticClassifier` compatibility.

        Returns
        -------
        bool
            True if the object is a `ProbabilisticClassifier` (has a "predict_proba" method ), False otherwise.
        """
        return hasattr(other, "predict_proba")


class ProbabilisticClassifier(metaclass=ProbabilisticClassifierMeta):
    """Base class for `ProbabilisticClassifier`.

    This base class defines the `ProbabilisticClassifier` interface, indicating that subclasses should have a
    "predict_proba" method.
    """

    pass


class ClustererMeta(type):
    """Metaclass for `Clusterer`.

    This metaclass is responsible for checking whether a class can be considered a `Clusterer`.
    A class is considered a `Clusterer` if it has a "fit_predict" method.
    """

    def __instancecheck__(self, other):
        """Checks if the provided object is a `Clusterer`.

        Parameters
        ----------
        self : ClustererMeta
            `ClustererMeta` class.
        other : object
            The object to check for `Clusterer` compatibility.

        Returns
        -------
        bool
            True if the object is a `Clusterer` (has a "fit_predict" method ), False otherwise.
        """
        return hasattr(other, "fit_predict")


class Clusterer(metaclass=ClustererMeta):
    """Base class for `Clusterer`.

    This base class defines the `Clusterer` interface, indicating that subclasses should have a "fit_predict" method.
    """

    pass


class OutlierModelMeta(type):
    """Metaclass for `OutlierModel`.

    This metaclass is responsible for checking whether a class can be considered an `OutlierModel`.
    A class is considered an `OutlierModel` if it is an instance of the `sklearn.base.OutlierMixin` class.
    """

    def __instancecheck__(self, other):
        """
        Check if the provided object is an `OutlierModel`.

        Parameters
        ----------
        self : OutlierModelMeta
            The `OutlierModelMeta` class.
        other : object
            The object to check for `OutlierModel` compatibility.

        Returns
        -------
        bool
            True if the object is an `OutlierModel` (an instance of "OutlierMixin"), False otherwise.
        """
        return isinstance(other, OutlierMixin)


class OutlierModel(metaclass=OutlierModelMeta):
    """Base class for `OutlierModel`.

    This base class defines the `OutlierModel` interface, indicating that subclasses should be instances of the
    "OutlierMixin" class.
    """

    pass
