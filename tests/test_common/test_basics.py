from sklego.common import as_list


def test_as_list_strings():
    assert as_list('test') == ['test']
    assert as_list(['test1', 'test2']) == ['test1', 'test2']


def test_as_list_ints():
    assert as_list(123) == [123]
    assert as_list([1, 2, 3]) == [1, 2, 3]


def test_as_list_other():

    def f():
        return 123

    assert as_list(f) == [f]
    assert as_list(range(1, 4)) == [1, 2, 3]
