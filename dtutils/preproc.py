import numpy as np
import copy


def np_features(f):
    if not isinstance(f, np.ndarray):
        return np.array(f)
    else:
        return f


def subsample_features(f, labels, n_samples):
    n_tot = len(f)
    if isinstance(n_samples, float) and n_samples < 1:
        n_samples = max(1., n_tot * n_samples)
        
    n_samples = int(n_samples)
    if n_samples > n_tot or n_samples <= 0:
        return f, labels
    
    random_subsample_idx = np.random.choice(n_tot, size=n_samples, replace=False)
    random_subsample_idx.sort()
    f_subsample = f[random_subsample_idx]
    l_subsample = [labels[i] for i in random_subsample_idx] if labels is not None else None
    return f_subsample, l_subsample


def center_features(f):
    f = copy.deepcopy(f)
    f -= f.mean(axis=0, keepdims=True)
    return f


def std_features(f):
    f = copy.deepcopy(f)
    f -= f.mean(axis=0, keepdims=True)
    f /= f.std(axis=0, keepdims=True)
    return f


def scale_features(f):
    f = copy.deepcopy(f)
    f -= f.min(axis=0, keepdims=True)
    f /= f.max(axis=0, keepdims=True)
    f *= 2
    f -= 1
    return f


def project_sphere_features(f):
    f = center_features(f)
    f /= np.sqrt((f**2).sum(axis=1, keepdims=True))
    return f
