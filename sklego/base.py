class ProbabilisticClassifierMeta(type):
    def __instancecheck__(self, other):
        return hasattr(other, 'predict_proba')


class ProbabilisticClassifier(metaclass=ProbabilisticClassifierMeta):
    pass
