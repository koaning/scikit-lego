# import numpy as np
# from sklego.outlier import GMMDetector


# def test_obvious_usecase():
#     X = np.concatenate([np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (1, 2))])
#     y = np.concatenate([np.zeros(100), np.ones(1)])
#     assert (GMMDetector().fit(X).predict(X) == y).all()
