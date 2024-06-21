"""Main datatour functionality implementation."""

import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from datatour.dtutils import examples
from datatour.dtutils.log import log
from datatour.dtutils.preproc import (
    center_features,
    np_features,
    project_sphere_features,
    scale_features,
    std_features,
    subsample_features,
)
from datatour.dtutils.rotations import gen_disp_ds


def fill_n_scaled(d):
    """Fills Z values to according to n values scaled [0, 1]."""
    key = "n_scaled"
    if key not in d:
        za = np.array(d["n"], dtype=np.float32)

        za -= za.min()
        za /= za.max()

        d[key] = list(za)
    return key


def fill_z_scaled(d):
    """Fills Z values to according to z values scaled [0, 1]."""
    key = "z_scaled"
    if key not in d:
        za = np.array(d["z"])

        za -= za.min()
        za /= za.max()

        d[key] = list(za)
    return key


def fill_z_scaled_inverted(d):
    """Fills Z values to according to z values scaled [1, 0] (i.e. inverted)."""
    key = "z_scaled_inv"

    k_zs = fill_z_scaled(d)
    if key not in d:
        za = np.array(d[k_zs])

        za = 1 - za

        d[key] = list(za)
    return key


def fill_z_scaled_exp(d):
    """Fills Z values to as `exp(-z_inv_scale/z_scale)`."""
    key = "z_scaled_exp"
    z_scale = 0.8

    k_zs = fill_z_scaled_inverted(d)
    if key not in d:
        za = np.array(d[k_zs])

        za = np.exp(-za / z_scale)

        d[key] = list(za)
    return key


def fill_z_order(d):
    """Fills scaled Z values to `exp(z_inv_scale/z_scale)`."""
    key = "z_order"

    k_zs = fill_z_scaled(d)
    if key not in d:
        za = np.array(d[k_zs])

        log("filling z_idx...", end=" ")
        za = np.argsort(za).astype(np.float32)
        za -= za.min()
        za /= za.max()
        log("Done")

        d[key] = list(za)
    return key


def fill_z_order_inverted(d):
    """Fills scaled Z values to `exp(z_inv_scale/z_scale)`."""
    key = "z_order_inv"

    k_zs = fill_z_order(d)
    if key not in d:
        za = np.array(d[k_zs])

        za = 1 - za

        d[key] = list(za)
    return key


def fill_z_order_exp(d):
    """Fills scaled Z values to `exp(-z_order_inv/z_scale)`."""
    key = "z_order_exp"
    z_scale = 0.8

    k_zs = fill_z_order_inverted(d)
    if key not in d:
        za = np.array(d[k_zs])

        za = np.exp(-za / z_scale)

        d[key] = list(za)
    return key


class DataTour:
    """DataTour class for data visualization."""

    def __init__(
        self,
        data=None,
        data_v=None,
        labels=None,
        rot_sampling=90,
        n_rot_vectors=5,
        n_subsample=500,
        norm="scale",
        example="cube",
        example_num_dim=4,
        generate_new_mtx=False,
    ):
        """
        Constructor for DataTour class.

        Args:
            data(array or str): 2d array [sample_idx, feature_idx] or None
            data_v(array): 2d array [sample_idx, vector_feature_idx] or
                           None - used for the quiver plot
            labels(array): 1d array [sample_idx] of class idx - used only for
                           display color coding with color='l' flag
            rot_sampling(int): number of samples per 2pi rotation
            n_rot_vectors(int): how many different random rotation axes
                                to sample
            n_subsample(int): sample at most n_subsample samples from the
                              feature list
            norm(str): feature normalization before projecting.
                       'sphere': project features on unit sphere
                       'scale': scaling [-1:1] (per feature)
                       'std': normalize (per feature)
                       'center': subtract mean (per feature)
            example(str): string or None, string containing sample dataset name:
                                'cube', 'table4d', 'ballshell', 'ballshell2',
                                'sphere', 'filledcube',
            example_num_dim(int): dimensionality of example to be generated,
                             if supported. default: 4


        """
        if data is None:
            data = self.sample_data(example, example_num_dim)

        self.features, self.features_v, self.labels = self.prepare_features(
            data=data, data_v=data_v, labels=labels, norm=norm, n_subsample=n_subsample
        )
        self.dict = gen_disp_ds(
            x=self.features,
            x_v=self.features_v,
            n_smpl_rot=rot_sampling,
            n_rot=n_rot_vectors,
            labels=self.labels,
            regenerate_mtx=generate_new_mtx,
        )
        fill_z_scaled_exp(self.dict)  # fill default drawing option
        fill_n_scaled(self.dict)  # fill default drawing option

        self.last_fig = None
        self.last_qfig = None

    @staticmethod
    def prepare_features(data, data_v, labels, norm, n_subsample):
        """
        Prepares features for visualization by normalizing and subsampling.

        Args:
            data(array): 2d array [sample_idx, feature_idx]
            data_v(array): 2d array [sample_idx, vector_feature_idx]
            labels(array): 1d array [sample_idx] of class idx
            norm(str): feature normalization before projecting.
                       'sphere': project features on unit sphere
                       'scale': scaling [-1:1] (per feature)
                       'std': normalize (per feature)
                       'center': subtract mean (per feature)
            n_subsample(int): sample at most n_subsample samples from the
                              feature list
        """
        features = np_features(data)
        features_v = np_features(data_v)

        features, features_v, labels = subsample_features(
            f=features, f_v=features_v, labels=labels, n_samples=n_subsample
        )

        if norm == "sphere":
            features, features_v = project_sphere_features(features, features_v)
        if norm == "scale":
            features, features_v = scale_features(features, features_v)
        if norm == "std":
            features, features_v = std_features(features, features_v)
        if norm == "center":
            features, features_v = center_features(features, features_v)

        return features, features_v, labels

    @staticmethod
    def sample_data(example, n_dim):
        """
        Samples example data.

        Args:
            example (str): string containing sample dataset name
                           'cube', 'table_4d', 'ball_shell', 'ball_shell2',
                           'sphere', and 'filled_cube' are supported
            n_dim (int): dimensionality of example to be generated

        Returns
        -------
            ndarray: example data

        """
        if example == "cube":
            return examples.cube(n_dim)
        if example == "table_4d":
            return examples.table_4d()
        if example == "ball_shell":
            return examples.ball_shell(n_dim=n_dim)
        if example == "sphere":
            return examples.sphere_equators(n_dim=n_dim)
        if example == "filled_cube":
            return examples.filled_cube(n_dim=n_dim)
        if example == "ball_shell2":
            return examples.filled_sphere(n_dim=n_dim)

    def display(
        self,
        color="n_scaled",
        size="z_scaled_exp",
        point_size=8,
        fig_size=800,
        cmap="jet",
    ):
        """
        Main function, displays the animated scatter plot.

        Args:
            color(str): None or one of self.dict keys, e.g.
                   'l' - labels or 'z_scaled',
                   'z_scaled_inv', 'z_scaled_exp', 'z_order', 'z_order_inv',
                   'z_order_exp', 'n_scaled'(default)
            size(str): None, or 'z_scaled', 'z_scaled_inv',
                   'z_scaled_exp'(default),
                   'z_order', 'z_order_inv', 'z_order_exp'
            point_size(int): max size of points on the scatter plot
            fig_size(int): width and height of the plot
            cmap(str): type of colormap for points colorscale
        """
        if (color is not None and "order" in color) or (
            size is not None and "order" in size
        ):
            fill_z_order_exp(self.dict)

        fig = px.scatter(
            self.dict,
            x="x",
            y="y",
            color=color,
            size=size,
            animation_frame="t",
            width=fig_size,
            height=fig_size,
            size_max=point_size,
            color_continuous_scale=cmap,
        )

        fig.update_xaxes(range=[-3, 3])
        fig.update_yaxes(range=[-3, 3])
        fig.update_layout(yaxis={"scaleanchor": "x"})
        fig.update_layout(transition={"duration": 5})
        fig.show()
        self.last_fig = fig

    def display_quiver(
        self,
        color="n_scaled",
        size="z_scaled_exp",
        point_size=12,
        angle=15 * np.pi / 180,
        uv_scale=1,
        arrow_scale=0.17,
        alpha=0.8,
        fig_size=800,
        cmap="jet",
    ):
        """
        Displays animated quiver plot.

        Args:
            color(str): None or one of self.dict keys, e.g. labels' or
                   'z_scaled', 'z_scaled_inv', 'z_scaled_exp', 'z_order',
                   'z_order_inv', 'z_order_exp', 'n_scaled'(default)
            size(str): None, or 'z_scaled', 'z_scaled_inv',
                    'z_scaled_exp'(default), 'z_order', 'z_order_inv',
                    'z_order_exp'
            point_size(int): max size of points on the scatter plot
            angle(float): angle between whiskers of the arrow in radians
            uv_scale(float): normalization factor for vector length
            arrow_scale(float): scale of the arrow head wrt whole arrow length
            alpha(float): opacity of the arrows
            fig_size(int): width and height of the plot
            cmap(str): type of colormap for points color scale
        """
        if (color is not None and "order" in color) or (
            size is not None and "order" in size
        ):
            fill_z_order_exp(self.dict)

        x = np.array(self.dict["x"])
        y = np.array(self.dict["y"])
        u = np.array(self.dict["u"])
        v = np.array(self.dict["v"])
        t = np.array(self.dict["t"])
        c = np.array(self.dict[color])
        s = np.array(self.dict[size])

        t_ids = sorted(set(t))

        opacity = alpha

        def get_at_t(selected_t_id):
            m = t == selected_t_id
            xs, ys, us, vs, ts, cs, ss = x[m], y[m], u[m], v[m], t[m], c[m], s[m]
            return xs, ys, us, vs, ts, cs, ss

        k = len(t_ids)

        frames = []
        for i, t_id in enumerate(t_ids):
            xt, yt, ut, vt, tt, ct, st = get_at_t(t_id)

            quiver_data = ff.create_quiver(
                xt,
                yt,
                ut,
                vt,
                scale=uv_scale,
                arrow_scale=arrow_scale,
                angle=angle,
                line={"width": 0.75, "color": "lightsteelblue"},
                opacity=opacity,
            ).data[0]

            if point_size == 0:
                scatter_data = None
            else:
                scatter_data = go.Scatter(
                    x=xt,
                    y=yt,
                    name="Location",
                    mode="markers",
                    marker={
                        "color": ct,
                        "colorscale": cmap,
                        "size": point_size * st,
                        "line": {"color": "white", "width": 0.5},
                        "cmin": 0,
                        "cmax": 1,
                        "colorbar": {
                            "thickness": 10,
                            "tickvals": [0, 1],
                            "ticktext": ["Far", "Close"],
                            "outlinewidth": 0,
                        },
                    },
                )

                # Adding frames
            data = (
                [quiver_data] if scatter_data is None else [quiver_data, scatter_data]
            )
            traces = [0] if scatter_data is None else [0, 1]

            frames.append({"name": i, "data": data, "traces": traces})

        layout = go.Layout(
            width=fig_size,
            height=fig_size,
            showlegend=False,
            hovermode="closest",
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "y": -0.1,
                    "x": 0,
                    "xanchor": "left",
                    "yanchor": "top",
                    "pad": {"t": 0, "r": 10},
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 5, "redraw": False},
                                    "transition": {"duration": 0},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 5, "redraw": False},
                                    "transition": {"duration": 0},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                },
            ],
        )

        fig = go.Figure(data=frames[0]["data"], frames=frames, layout=layout)

        # Adding a slider
        sliders = [
            {
                "yanchor": "top",
                "xanchor": "left",
                "active": 1,
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Steps: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 200, "easing": "linear"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.15,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [i],
                            {
                                "frame": {
                                    "duration": 5,
                                    "easing": "linear",
                                    "redraw": False,
                                },
                                "transition": {"duration": 0, "easing": "linear"},
                            },
                        ],
                        "label": i,
                        "method": "animate",
                    }
                    for i in range(k)
                ],
            }
        ]

        fig["layout"].update(sliders=sliders)

        fig.update_xaxes(range=[-3, 3])
        fig.update_yaxes(range=[-3, 3])
        fig.update_layout(yaxis={"scaleanchor": "x"})

        fig.show()

        self.last_qfig = fig
