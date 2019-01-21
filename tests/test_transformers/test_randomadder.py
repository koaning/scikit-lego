from skblocks.transformers import RandomAdder


def test_shape_does_not_change(random_xy_dataset):
    X, y = random_xy_dataset
    assert RandomAdder().fit(X, y).transform(X).shape == X.shape
