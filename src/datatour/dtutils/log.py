_VERB_ERR = 0  # 0 - nothing
_VERB_WRN = 1  # 1 - warnings
_VERB_INF = 2  # 2 - info

_VERBOSITY = _VERB_INF


def log(message, verbosity=_VERB_INF, end='\n'):
    if verbosity <= _VERBOSITY:
        print(message, end=end)
