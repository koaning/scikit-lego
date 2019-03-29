

def as_list(val):
    """
    Helper function, always returns a list of the input value.

    :param val: the input value.
    :returns: the input value as a list.

    :Example:

    >>> as_list('test')
    ['test']

    >>> as_list(['test1', 'test2'])
    ['test1', 'test2']
    """
    treat_single_value = (str)

    if isinstance(val, treat_single_value):
        return [val]

    if hasattr(val, '__iter__'):
        return list(val)

    return [val]
