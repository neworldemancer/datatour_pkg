"""rotation utilities in the high dimensional space."""

import os
import pickle
import tempfile
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy import sparse

_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "GrandTour_RM_cache")
os.makedirs(_CACHE_ROOT, exist_ok=True)


def get_rand_rot(
    n: int, i: int, j: int, phi: Optional[float] = None
) -> sparse.coo_matrix:
    """Generate a rotation matrix in the i-j plane of an n-dimensional space."""
    phi = phi or np.random.uniform(0, np.pi * 2)
    c = np.cos(phi)
    s = np.sin(phi)

    i_arr = [i, i, j, j]
    j_arr = [i, j, i, j]
    v_arr = [c, -s, s, c]
    for diag_idx in range(n):
        if diag_idx != i and diag_idx != j:
            i_arr.append(diag_idx)
            j_arr.append(diag_idx)
            v_arr.append(1)

    rot_mtx = sparse.coo_matrix((v_arr, (i_arr, j_arr)), shape=(n, n))
    return rot_mtx


def get_rand_rot_rev(n: int) -> sparse.coo_matrix:
    """Generate a random rotation matrix in the n-dimensional space."""
    rot_matrices = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            r = get_rand_rot(n, i, j)
            rot_matrices.append(r)

    r_prod = rot_matrices[0]
    for r in rot_matrices[1:]:
        r_prod = r @ r_prod

    return r_prod


def get_rand_rot_rev_d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random rotation+inv matrix in the n-dimensional space."""
    rot_mtx_row_prod = []
    for i in range(n - 1):
        print(f"i={i}", end="\r")
        r_row: sparse.coo_matrix = None
        for j in range(i + 1, n):
            r = get_rand_rot(n, i, j)
            if j == i + 1:
                r_row = r
            else:
                r_row = r @ r_row
        rot_mtx_row_prod.append(np.array(r_row.todense()))

    r_prod = rot_mtx_row_prod[0]
    for r in rot_mtx_row_prod[1:]:
        r_prod = r @ r_prod

    rr_prod = np.linalg.inv(r_prod)

    return r_prod, rr_prod


def get_r_phi(n: int, phi: float) -> np.ndarray:
    """Get a rotation matrix in the xy plane of the n-d space with angle phi."""
    return np.array(get_rand_rot(n, 0, 1, phi).todense())


def get_projection(
    n: int,
    r_rs: np.ndarray,
    rr_rs: np.ndarray,
    r_d: np.ndarray,
    v: np.ndarray,
    phi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a point in the n-dimensional space to the display 2D plane."""
    r_p = get_r_phi(n, phi)
    v_d = r_d @ (rr_rs @ (r_p @ (r_rs @ v)))

    v_p = v_d[:2]
    z = v_d[2:]
    n = len(v_d)
    a = np.ones((n - 2, 1))
    z = z * a
    z = z.sum(axis=0)
    return v_p, z


def get_cache_file(n_dim: int) -> str:
    """Get the cache file name for n_dim rotations."""
    os.makedirs(_CACHE_ROOT, exist_ok=True)
    return os.path.join(_CACHE_ROOT, f"rot_cache_{n_dim}.pckl")


def load_pckl(file_name: str) -> dict:
    """Load a pickle file."""
    with open(file_name, "rb") as f:
        data: dict = pickle.load(f)
    return data


def save_pckl(d: dict, file_name: str) -> None:
    """Save an object to a pickle file."""
    with open(file_name, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cached_rot_mtxs(n_dim: int) -> dict:
    """Load the cached n_dim rotation matrices."""
    fn = get_cache_file(n_dim)
    if os.path.exists(fn):
        cache = load_pckl(fn)
    else:
        cache = {"version": "0.1", "n_dim": n_dim, "rand_rot_mtx_collection": []}
    return cache


def save_cached_rot_mtxs(cache: dict) -> None:
    """Cache the rotation matrices."""
    n_dim = cache["n_dim"]

    fn = get_cache_file(n_dim)
    save_pckl(cache, fn)


def gen_cache_rand_rot(
    n_dim: int, n_samples: int, regenerate: bool = False
) -> List[np.ndarray]:
    """Generate a collection of random rotation matrices in the n_dim space."""
    # get collection_file_name from n_dim
    # read cache info
    cache = load_cached_rot_mtxs(n_dim)
    cached_mtx = cache["rand_rot_mtx_collection"]
    n_cached = len(cached_mtx)

    # generate missing or all
    n_new = n_samples if regenerate else max(0, n_samples - n_cached)
    print(
        f"Found {n_cached} mtx for {n_dim}-dimensional rotations in cache."
        f" {n_new} will be generated"
    )
    new_mtx = [get_rand_rot_rev_d(n_dim) for i in range(n_new)]

    # add to collection
    if n_new:
        cached_mtx.extend(new_mtx)

    if regenerate:
        res = new_mtx
    else:
        n_cached = len(cached_mtx)
        mask = np.random.choice(n_cached, size=n_samples, replace=False)
        print(mask)
        res = [cached_mtx[idx] for idx in mask]

    # save collection
    if n_new:
        save_cached_rot_mtxs(cache)

    return res


def gen_disp_ds(
    x: np.ndarray,
    n_smpl_rot: int = 100,
    n_rot: int = 1,
    labels: Optional[Iterable] = None,
    regenerate_mtx: bool = False,
    x_v: np.ndarray = None,
) -> dict:
    """Generate a dataset for the display."""
    d: dict = {"t": [], "x": [], "y": [], "z": [], "l": [], "n": []}

    n_smpl, n = x.shape

    if x_v is not None:
        n_smpl_v, n_v = x_v.shape
        assert n_smpl_v == n_smpl
        assert n_v == n

        d["u"] = []
        d["v"] = []

    rot_mtxs = gen_cache_rand_rot(n, n_rot + 1, regenerate=regenerate_mtx)

    r_d, _ = rot_mtxs[0]
    rot_mtxs_rot = rot_mtxs[1:]

    if labels is not None:
        lbls = np.array(labels).astype(np.float32)
        lbls -= lbls.min()
        m = lbls.max()
        if m != 0:
            lbls /= m
        lbls = list(lbls)
    else:
        lbls = [0.0] * n_smpl

    for rot in range(n_rot):
        r_rs, rr_rs = rot_mtxs_rot[rot]

        print(f"rot={rot}")
        for t in np.linspace(
            np.pi * 2 * rot, np.pi * 2 * (rot + 1), n_smpl_rot, endpoint=False
        ):
            v_p, z = get_projection(n, r_rs, rr_rs, r_d, x.T, t)

            d["t"].extend([t] * n_smpl)
            d["z"].extend(list(z))
            d["x"].extend(list(v_p[0]))
            d["y"].extend(list(v_p[1]))
            d["l"].extend(lbls)
            d["n"].extend(list(range(n_smpl)))

            if x_v is not None:
                v_p_v, z_v = get_projection(n, r_rs, rr_rs, r_d, x_v.T, t)

                d["u"].extend(list(v_p_v[0]))
                d["v"].extend(list(v_p_v[1]))

    return d
