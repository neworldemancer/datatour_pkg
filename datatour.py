import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
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
                 data_v=None,
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
            data_v(array): 2d array [sample_idx, vector_feature_idx] or None - used for the quiver plot
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

        self.features, self.features_v, self.labels = self.prepare_features(data=data, data_v=data_v, labels=labels, norm=norm, n_subsample=n_subsample)
        self.dict = gen_disp_ds(x=self.features,
                                x_v=self.features_v,
                                n_smpl_rot=rot_sampling,
                                n_rot=n_rot_vectors,
                                labels=self.labels,
                                regenerate_mtx=generate_new_mtx)
        fill_z_scaled_exp(self.dict)  # fill default drawing option
        fill_n_scaled(self.dict)  # fill default drawing option

    @staticmethod
    def prepare_features(data, data_v, labels, norm, n_subsample):
        features = np_features(data)
        features_v = np_features(data_v)

        features, features_v, labels = subsample_features(f=features, f_v=features_v, labels=labels, n_samples=n_subsample)

        if norm == 'sphere':
            features, features_v = project_sphere_features(features, features_v)
        if norm == 'scale':
            features, features_v = scale_features(features, features_v)
        if norm == 'std':
            features, features_v = std_features(features, features_v)
        if norm == 'center':
            features, features_v = center_features(features, features_v)

        return features, features_v, labels

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

    def display_quiver(self,
                color='n_scaled',
                size='z_scaled_exp',
                point_size=8,
                angle = 15*np.pi/180,
                uvscale = 1,
                arrow_scale = 0.5,
                fig_size=800,
                cmap='jet'):
        """
        splits inputs
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

        x = np.array(self.dict['x'])
        y = np.array(self.dict['y'])
        u = np.array(self.dict['u'])
        v = np.array(self.dict['v'])
        t = np.array(self.dict['t'])
        c = np.array(self.dict[color]) 
        s = np.array(self.dict[size])
        
        t_ids = set(t)
        
        def get_at_t(t_id):
            m = t==t_id
            xt, yt, ut, vt, tt, ct, st = x[m], y[m], u[m], v[m], t[m], c[m], s[m]
            return xt, yt, ut, vt, tt, ct, st
            
        
        xt, yt, ut, vt, tt, ct, st = get_at_t(t[0])

        fig = ff.create_quiver(xt, yt, ut, vt,
                               scale=uvscale,
                               arrow_scale=arrow_scale,
                               angle=angle, 
                               line=dict(width=1.25, color='#8f180b'))


        k = len(t_ids)

        frames = []
        for i, t_id in enumerate(t_ids):
            xt, yt, ut, vt, tt, ct, st = get_at_t(t_id)

            figaux = ff.create_quiver(xt, yt, ut, vt, 
                                      scale=uvscale,
                                      arrow_scale=arrow_scale,
                                      angle=angle)
            trace = go.Scatter(x=xt, y=yt,
                                        name='Location',
                                        mode='markers',
                                        marker=dict(color=ct, 
                                                    size=point_size*st,
                                                    line=dict(
                                                    color='DarkSlateGrey',
                                                    width=0.5)
                                                   )
                                        )    

            # Adding frames
            frames.append(dict(name=i,data=[figaux.data[0], trace],
                               traces = [0, 1])
                         )


        layout = go.Layout(width=fig_size,
                           height=fig_size,
                           showlegend=False,
                           hovermode='closest',
                           updatemenus=[dict(type='buttons', showactive=False,
                                           y=-.1,
                                           x=0,
                                           xanchor='left',
                                           yanchor='top',
                                           pad=dict(t=0, r=10),
                                           buttons=[dict(label='Play',
                                                       method='animate',
                                                       args=[None, 
                                                               dict(frame=dict(duration=1, redraw=False), 
                                                                   transition=dict(duration=0),
                                                                   fromcurrent=True,
                                                                   mode='immediate'
                                                                           )
                                                       ]),
                                                   dict(label='Pause', # https://github.com/plotly/plotly.js/issues/1221 / https://plotly.com/python/animations/#adding-control-buttons-to-animations
                                                       method='animate',
                                                       args=[[None],
                                                               dict(frame=dict(duration=0, redraw=False), 
                                                                transition=dict(duration=0),
                                                                fromcurrent=True,
                                                                mode='immediate' )
                                                    ])
                                                ])
                                    ])

        fig = go.Figure(data=frames[0]['data'], frames=frames, layout=layout)

        # Adding a slider
        sliders = [{
                'yanchor': 'top',
                'xanchor': 'left', 
                'active': 1,
                'currentvalue': {'font': {'size': 16}, 'prefix': 'Steps: ', 'visible': True, 'xanchor': 'right'},
                'transition': {'duration': 200, 'easing': 'linear'},
                'pad': {'b': 10, 't': 50}, 
                'len': 0.9, 'x': 0.15, 'y': 0, 
                'steps': [{'args': [[i], {'frame': {'duration': 5, 'easing': 'linear', 'redraw': False},
                                            'transition': {'duration': 0, 'easing': 'linear'}}], 
                            'label': i, 'method': 'animate'} for i in range(k)       
                        ]}]

        fig['layout'].update(sliders=sliders)

        fig.update_xaxes(range=[-3, 3])
        fig.update_yaxes(range=[-3, 3])
        fig.update_layout(yaxis=dict(scaleanchor="x"))

        fig.show()
        #return fig
        