"""Collection of example datasets for testing and demonstration purposes."""

from typing import Generator

import numpy as np


def cube_5d() -> np.ndarray:
    """Vertex coordinates of a 5-dimensional cube."""
    box = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    return box


def cube(n_dim: int) -> np.ndarray:
    """Vertex coordinates of a n-dimensional cube. n_dim <= 5."""
    d = n_dim
    if d > 5:
        raise ValueError(f"{d}D-cube not implemented")
    n = 2**d
    box5d = cube_5d()
    box = box5d[:n, -d:]
    return box


def table_4d() -> np.ndarray:
    """Vertex coordinates of a 4-dimensional table object."""
    box = np.array(
        [
            [0, 12, 0, 0],
            [0, 12, 0, 12],
            [0, 12, 12, 0],
            [0, 12, 12, 12],
            [0, 14, 0, 0],
            [0, 14, 0, 12],
            [0, 14, 12, 0],
            [0, 14, 12, 12],
            [0, 0, 4, 4],
            [0, 0, 4, 6],
            [0, 0, 6, 4],
            [0, 0, 6, 6],
            [0, 12, 4, 4],
            [0, 12, 4, 6],
            [0, 12, 6, 4],
            [0, 12, 6, 6],
            [1, 12, 0, 0],
            [1, 12, 0, 12],
            [1, 12, 12, 0],
            [1, 12, 12, 12],
            [1, 14, 0, 0],
            [1, 14, 0, 12],
            [1, 14, 12, 0],
            [1, 14, 12, 12],
            [1, 0, 4, 4],
            [1, 0, 4, 6],
            [1, 0, 6, 4],
            [1, 0, 6, 6],
            [1, 12, 4, 4],
            [1, 12, 4, 6],
            [1, 12, 6, 4],
            [1, 12, 6, 6],
        ],
        dtype=np.float32,
    )
    return box


def ball_shell(n_dim: int = 6, n_points: int = 1000) -> np.ndarray:
    """Points distributed in a shell of a ball in n_dim dimensions."""
    box = np.random.uniform(-5, 5, size=(n_points, n_dim))
    r = np.sqrt((box**2).sum(axis=1))
    mask = (r > 4) * (r < 5)
    box = box[mask] / r[mask][:, np.newaxis]
    return box


def sphere_equators(n_dim: int = 4, n_phi: int = 60) -> np.ndarray:
    """n_phi = number of uniformly spaced angles on the equator."""
    p = np.linspace(0, np.pi * 2, n_phi)
    c = np.cos(p)
    s = np.sin(p)
    circle = np.stack([c, s] + [[0.0] * n_phi] * (n_dim - 2), axis=1)

    def permute(arr: list) -> Generator[list, None, None]:
        length = len(arr)
        if length <= 1:
            yield arr
        else:
            for n in range(0, length):
                for end in permute(arr[:n] + arr[n + 1 :]):
                    yield [arr[n], *end]

    def ordered_ax01_perm(perm: list) -> bool:
        perm_arr = np.array(perm)
        return int(np.argmax(perm_arr == 0)) < int(np.argmax(perm_arr == 1))

    axes = list(range(n_dim))
    box = [circle[:, perm] for perm in permute(axes) if ordered_ax01_perm(perm)]
    box = np.concatenate(box)

    return box


def filled_cube(n_dim: int = 4, n_points: int = 1000) -> np.ndarray:
    """Points uniformly distributed in a n-dimensional cube."""
    box = np.random.uniform(-10, 10, size=(n_points, n_dim))
    box = np.round(box)

    box = box.astype(np.int32)
    box = np.array(list({tuple(sample) for sample in box}), dtype=np.float32)
    box /= 10
    return box


def filled_sphere(n_dim: int = 4, n_points: int = 1000) -> np.ndarray:
    """Points uniformly distributed in a n-dimensional sphere except center."""
    box = filled_cube(n_dim=n_dim, n_points=n_points)
    r = np.sqrt((box**2).sum(axis=1))
    mask = (r > 0.5) * (r < 1)

    box = box[mask]
    return box
