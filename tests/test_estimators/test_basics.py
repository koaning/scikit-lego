from sklego.dummy import RandomRegressor


def test_shape_regression(random_xy_dataset_regr):
    for estimator in [RandomRegressor]:
        X, y = random_xy_dataset_regr
        assert estimator().fit(X, y).predict(X).shape[0] == y.shape[0]
