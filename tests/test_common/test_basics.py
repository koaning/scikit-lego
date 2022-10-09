import pytest
from sklego.common import as_list, sliding_window


def test_as_list_strings():
    assert as_list("test") == ["test"]
    assert as_list(["test1", "test2"]) == ["test1", "test2"]


def test_as_list_ints():
    assert as_list(123) == [123]
    assert as_list([1, 2, 3]) == [1, 2, 3]


def test_as_list_other():
    def f():
        return 123

    assert as_list(f) == [f]
    assert as_list(range(1, 4)) == [1, 2, 3]


@pytest.mark.parametrize(
    "sequence, window_size, step_size",
    [([1, 2, 3, 4, 5], 2, 1), (["a", "b", "c", "d", "e"], 3, 2)],
)
def test_sliding_window(sequence, window_size, step_size):
    windows = list(sliding_window(sequence, window_size, step_size))
    assert windows[0] == sequence[:window_size]
    assert len(windows[0]) == window_size
    assert windows[1][0] == sequence[step_size]
