"""collection of functions to preprocess features."""

import copy

import numpy as np


def np_features(f):
    """Converts f to a numpy array if it is not already."""
    if f is None:
        return None

    if not isinstance(f, np.ndarray):
        return np.array(f)
    else:
        return f


def subsample_features(f, labels, n_samples, f_v=None):
    """Subsample features and labels."""
    n_tot = len(f)
    if isinstance(n_samples, float) and n_samples < 1:
        n_samples = max(1.0, n_tot * n_samples)

    n_samples = int(n_samples)
    if n_samples > n_tot or n_samples <= 0:
        return f, f_v, labels

    random_subsample_idx = np.random.choice(n_tot, size=n_samples, replace=False)
    random_subsample_idx.sort()

    f_subsample = f[random_subsample_idx]
    f_v_subsample = None if f_v is None else f_v[random_subsample_idx]
    l_subsample = (
        [labels[i] for i in random_subsample_idx] if labels is not None else None
    )
    return f_subsample, f_v_subsample, l_subsample


def center_features(f, f_v):
    """Center features by subtracting the mean."""
    f = copy.deepcopy(f)
    f -= f.mean(axis=0, keepdims=True)
    return f, f_v


def std_features(f, f_v):
    """Standardize features."""
    f = copy.deepcopy(f)
    f -= f.mean(axis=0, keepdims=True)
    norm = f.std(axis=0, keepdims=True)
    f /= norm
    if f_v is not None:
        f_v /= norm
    return f, f_v


def scale_features(f, f_v):
    """Scale features to [-1, 1]."""
    f = copy.deepcopy(f)
    f -= f.min(axis=0, keepdims=True)

    norm = f.max(axis=0, keepdims=True) / 2

    f /= norm
    f -= 1

    if f_v is not None:
        f_v /= norm
    return f, f_v


def project_sphere_features(f, f_v):
    """Project features to the surface of the unit sphere."""
    f, f_v = center_features(f, f_v)
    f /= np.sqrt((f**2).sum(axis=1, keepdims=True))
    return f, f_v
