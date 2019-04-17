from sklego.datasets import load_chicken


def test_chickweight():
    df = load_chicken()
    assert df.shape == (578, 4)
