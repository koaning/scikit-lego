from sklego.datasets import load_chicken


def test_chickweight1():
    X, y = load_chicken()
    assert X.shape == (578, 3)
    assert y.shape[0] == 578


def test_chickweight2():
    df = load_chicken(give_pandas=True)
    assert df.shape == (578, 4)
