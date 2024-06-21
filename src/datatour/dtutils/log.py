"""Simple logging utility for the DataTour package."""

_VERB_ERR = 0  # 0 - nothing
_VERB_WRN = 1  # 1 - warnings
_VERB_INF = 2  # 2 - info

_VERBOSITY = _VERB_INF


def log(message: str, verbosity: int = _VERB_INF, end: str = "\n") -> None:
    """Simple log: Print message according to verbosity level."""
    if verbosity <= _VERBOSITY:
        print(message, end=end)
