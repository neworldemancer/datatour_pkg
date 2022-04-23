import numpy as np
import plotly.express as px

from dtutils.log import *
from dtutils.rotations import gen_disp_ds
from dtutils.preproc import *
from dtutils import examples


def fill_n_scaled(d):
    """Fills scaled Z values to [0, 1]"""
    key = 'n_scaled'
    if key not in d:
        za = np.array(d['n'], dtype=np.float32)

        za -= za.min()
        za /= za.max()

        d[key] = list(za)
    return key


def fill_z_scaled(d):
    """Fills scaled Z values to [0, 1]"""
    key = 'z_scaled'
    if key not in d:
        za = np.array(d['z'])

        za -= za.min()
        za /= za.max()

        d[key] = list(za)
    return key


def fill_z_scaled_inverted(d):
    """Fills scaled Z values to [1, 0]"""
    key = 'z_scaled_inv'

    k_zs = fill_z_scaled(d)
    if key not in d:
        za = np.array(d[k_zs])

        za = 1 - za

        d[key] = list(za)
    return key


def fill_z_scaled_exp(d):
    """Fills scaled Z values to `exp(-z_inv_scale/z_scale)`"""
    key = 'z_scaled_exp'
    z_scale = 0.8

    k_zs = fill_z_scaled_inverted(d)
    if key not in d:
        za = np.array(d[k_zs])

        za = np.exp(-za / z_scale)

        d[key] = list(za)
    return key


def fill_z_order(d):
    """Fills scaled Z values to `exp(z_inv_scale/z_scale)`"""
    key = 'z_order'

    k_zs = fill_z_scaled(d)
    if key not in d:
        za = np.array(d[k_zs])

        log('filling z_idx...', end=' ')
        za = np.argsort(za).astype(np.float32)
        za -= za.min()
        za /= za.max()
        log('Done')

        d[key] = list(za)
    return key


def fill_z_order_inverted(d):
    """Fills scaled Z values to `exp(z_inv_scale/z_scale)`"""
    key = 'z_order_inv'

    k_zs = fill_z_order(d)
    if key not in d:
        za = np.array(d[k_zs])

        za = 1 - za

        d[key] = list(za)
    return key


def fill_z_order_exp(d):
    """Fills scaled Z values to `exp(-z_order_inv/z_scale)`"""
    key = 'z_order_exp'
    z_scale = 0.8

    k_zs = fill_z_order_inverted(d)
    if key not in d:
        za = np.array(d[k_zs])

        za = np.exp(-za / z_scale)

        d[key] = list(za)
    return key


class DataTour:
    def __init__(self, data=None,
                 labels=None,
                 rot_sampling=90,
                 n_rot_vectors=5,
                 n_subsample=500,
                 norm='scale',
                 example='cube',
                 example_nd=4,
                 generate_new_mtx=False
                 ):
        """
        Args:
            data(array or str): 2d array [sample_idx, feature_idx] or None
            rot_sampling(int): number of samples per 2pi rotation
            n_rot_vectors(int): how many different random rotation axes to sample
            n_subsample(int): sample at most n_subsample samples from the feature list
            norm(str): feature normalization before projecting. 
                       'sphere': project features on unit sphere
                       'scale': scaling [-1:1] (per feature)
                       'std': normalize (per feature)
                       'center': subtract mean (per feature)
            example(str): string or None, string containing sample dataset name:
                                'cube', 'table4d', 'ballshell', 'ballshell2', 'sphere', 'filledcube', 
            example_nd(int): dimensionality of example to be generated, if supported. default: 4


        """
        if data is None:
            data = self.sample_data(example, example_nd)

        self.features, self.labels = self.prepare_features(data, labels, norm, n_subsample)
        self.dict = gen_disp_ds(self.features,
                                n_smpl_rot=rot_sampling,
                                n_rot=n_rot_vectors,
                                labels=self.labels,
                                regenerate_mtx=generate_new_mtx)
        fill_z_scaled_exp(self.dict)  # fill default drawing option
        fill_n_scaled(self.dict)  # fill default drawing option

    @staticmethod
    def prepare_features(data, labels, norm, n_subsample):
        features = np_features(data)

        features, labels = subsample_features(features, labels, n_subsample)

        if norm == 'sphere':
            features = project_sphere_features(features)
        if norm == 'scale':
            features = scale_features(features)
        if norm == 'std':
            features = std_features(features)
        if norm == 'center':
            features = center_features(features)

        return features, labels

    @staticmethod
    def sample_data(example, n_dim):
        if example == 'cube':
            return examples.cube(n_dim)
        if example == 'table4d':
            return examples.table_4d()
        if example == 'ballshell':
            return examples.ball_shell(n_dim=n_dim)
        if example == 'sphere':
            return examples.sphere_equators(n_dim=n_dim)
        if example == 'filledcube':
            return examples.filled_cube(n_dim=n_dim)
        if example == 'ballshell2':
            return examples.filled_sphere(n_dim=n_dim)

    def display(self,
                color='n_scaled',
                size='z_scaled_exp',
                point_size=8,
                fig_size=800,
                cmap='jet'):
        """
        Args:
            color(str): None or one of self.dict keys, e.g. labels' or 'z_scaled',
                   'z_scaled_inv', 'z_scaled_exp', 'z_order', 'z_order_inv', 'z_order_exp',
                   'n_scaled'(default)
            size(str): None, or 'z_scaled', 'z_scaled_inv', 'z_scaled_exp'(default),
                   'z_order', 'z_order_inv', 'z_order_exp'
            point_size(int): max size of points on the scatter plot
            fig_size(int): width and height of the plot
            cmap(str): type of colormap for points colorscale
        """
        if (color is not None and 'order' in color) or (size is not None and 'order' in size):
            fill_z_order_exp(self.dict)

        fig = px.scatter(self.dict, x='x', y='y', color=color, size=size,
                         animation_frame='t',
                         width=fig_size, height=fig_size,
                         size_max=point_size,
                         color_continuous_scale=cmap
                         )

        fig.update_xaxes(range=[-3, 3])
        fig.update_yaxes(range=[-3, 3])
        fig.update_layout(yaxis=dict(scaleanchor="x"))
        fig.update_layout(transition={'duration': 5})
        fig.show()
