from collections.abc import Mapping, MutableMapping
from .path import path_exists_or_is_creatable, dir_exists_or_is_creatable
from .popen_timeout import PopenTimeout, PopenStorePID
from .unicode_symbols import SYMBOLS
import re


def perfect_square(val):
    import math

    root = math.sqrt(val)
    if int(root + 0.5) ** 2 == val:
        return True
    else:
        return False


def grouped_ordering(symbols):
    order = []
    uniqs = []
    for symbol in symbols:
        if symbol not in uniqs:
            uniqs.append(symbol)
    for uniq in uniqs:
        for i, symbol in enumerate(symbols):
            if uniq == symbol:
                order.append(i)
    return order


def recursive_dict_update(dict_to, dict_from):
    """Iterate through a dict, updating items recursively from a second
    dictionary
    >>> d1 = {'test': {'test_val': 3}}
    >>> d2 = {'test': {'test_val': 5}, 'other': 3}
    >>> recursive_dict_update(d1, d2)
    {'test': {'test_val': 5}, 'other': 3}
    """
    for key, val in dict_from.items():
        if isinstance(val, Mapping):
            dict_to[key] = recursive_dict_update(dict_to.get(key, {}), val)
        else:
            dict_to[key] = val
    return dict_to


def nested_dict_delete(root, key, sep="."):
    """Iterate through a dict, deleting items
    recursively based on a key
    >>> d1 = {'test': {'test_val': 3}}
    >>> d2 = {'test': {'test_val': 5, 'test_val_2': 7}, 'other': 3}
    >>> nested_dict_delete(d1, 'test.test_val')
    >>> d1
    {}
    >>> nested_dict_delete(d2, 'test.test_val')
    >>> d2
    {'test': {'test_val_2': 7}, 'other': 3}
    """

    levels = key.split(sep)
    level_key = levels[0]
    if level_key in root:
        if isinstance(root[level_key], MutableMapping):
            nested_dict_delete(root[level_key], sep.join(levels[1:]), sep=sep)
            if not root[level_key]:
                del root[level_key]
        else:
            del root[level_key]
    else:
        raise KeyError


def molecular_formula(labels, order="alphabetical"):
    """Return molecular formula from a list of atom labels
    Does not check if the element labels are valid at all.
    >>> molecular_formula(['H', 'C', 'H', 'H', 'O'])
    'CH3O'
    >>> molecular_formula(['H', 'h', 'H', 'H', 'O'])
    'H4O'
    """
    from collections import Counter

    counter = Counter(map(str.upper, labels))  # ensure uppercase first
    ordered = sorted(counter.items(), key=lambda x: x[0])
    return "".join("{}{}".format(e, n if n > 1 else "") for e, n in ordered)


__all__ = [
    "path_exists_or_is_creatable",
    "dir_exists_or_is_creatable",
    "PopenTimeout",
    "PopenStorePID",
    "SYMBOLS",
    "perfect_square",
]
