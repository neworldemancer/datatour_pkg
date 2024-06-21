import pytest

import datatour
from datatour import datatour as dt


def test_something():
    print(datatour.__version__)
    pass


@pytest.mark.parametrize("n_dim", [2, 3, 4, 5, 6])
def test_example_dims(n_dim):
    """ensure dimensions are correct ."""
    if n_dim <= 5:
        cube = dt.DataTour.sample_data("cube", n_dim=n_dim)
        n = 2**n_dim
        assert cube.shape == (n, n_dim), (
            f"cube_{n_dim}d should have shape " f"({n}, {n_dim})"
        )

    else:
        with pytest.raises(ValueError):
            dt.DataTour.sample_data("cube", n_dim=n_dim)
