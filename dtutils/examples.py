import numpy as np


def cube_3d():
    box = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],

    ], dtype=np.float32)
    return box


def cube_4d():
    box = np.array([
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

    ], dtype=np.float32)
    return box


def cube(n_dim):
    if n_dim == 3:
        return cube_3d()
    if n_dim == 4:
        return cube_4d()
    raise ValueError(f'{n_dim}D-cube not implemented')


def table_4d():
    box = np.array([
        [0, 12,  0,  0],
        [0, 12,  0, 12],
        [0, 12, 12,  0],
        [0, 12, 12, 12],
        [0, 14,  0,  0],
        [0, 14,  0, 12],
        [0, 14, 12,  0],
        [0, 14, 12, 12],
        [0,  0,  4,  4],
        [0,  0,  4,  6],
        [0,  0,  6,  4],
        [0,  0,  6,  6],
        [0, 12,  4,  4],
        [0, 12,  4,  6],
        [0, 12,  6,  4],
        [0, 12,  6,  6],

        [1, 12,  0,  0],
        [1, 12,  0, 12],
        [1, 12, 12,  0],
        [1, 12, 12, 12],
        [1, 14,  0,  0],
        [1, 14,  0, 12],
        [1, 14, 12,  0],
        [1, 14, 12, 12],
        [1,  0,  4,  4],
        [1,  0,  4,  6],
        [1,  0,  6,  4],
        [1,  0,  6,  6],
        [1, 12,  4,  4],
        [1, 12,  4,  6],
        [1, 12,  6,  4],
        [1, 12,  6,  6],
    ], dtype=np.float32)
    return box


def ball_shell(n_dim=6, n_points=1000):
    box = np.random.uniform(-5, 5, size=(n_points, n_dim))
    r = np.sqrt((box**2).sum(axis=1))
    mask = (r > 4) * (r < 5)
    box = box[mask] / r[mask][:, np.newaxis]
    return box


def sphere_equators(n_dim=4, n_phi=60):
    """
    n_phi = number of uniformly spaced angles on the
    """
    p = np.linspace(0, np.pi*2, n_phi)
    c = np.cos(p)
    s = np.sin(p)
    circle = np.stack([c, s]+[[0.]*n_phi]*(n_dim-2), axis=1)
    
    def permute(arr):
        length = len(arr)
        if length <= 1:
            yield arr
        else:
            for n in range(0, length):
                for end in permute(arr[:n] + arr[n+1:]):
                    yield [arr[n]] + end
                    
    def ordered_ax01_perm(perm):
        perm_arr = np.array(perm)
        return np.argmax(perm_arr == 0) < np.argmax(perm_arr == 1)
    
    axes = list(range(n_dim))
    box = [circle[:, perm] for perm in permute(axes) if ordered_ax01_perm(perm)]
    box = np.concatenate(box)

    return box


def filled_cube(n_dim=4, n_points=1000):
    box = np.random.uniform(-10, 10, size=(n_points, n_dim))
    box = np.round(box)
    
    box = box.astype(np.int32)
    box = np.array(list(set([tuple(sample) for sample in box])), dtype=np.float32)
    box /= 10
    return box


def filled_sphere(n_dim=4, n_points=1000):
    box = filled_cube(n_dim=n_dim, n_points=n_points)
    r = np.sqrt((box**2).sum(axis=1))
    mask = (r > 0.5) * (r < 1)

    box = box[mask]
    return box    
