from sklego.transformers import RandomAdder


def test_shape_regression(random_xy_dataset_regr):
    for transformer in [RandomAdder]:
        X, y = random_xy_dataset_regr
        assert transformer().fit(X, y).transform(X).shape == X.shape


def test_shape_classification(random_xy_dataset_clf):
    for transformer in [RandomAdder]:
        X, y = random_xy_dataset_clf
        assert transformer().fit(X, y).transform(X).shape == X.shape
