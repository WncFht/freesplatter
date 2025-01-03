import os

from PIL import Image
import plotly.graph_objects as go
import numpy as np


def calc_cam_cone_pts_3d(c2w, fov_deg, zoom = 1.0):

    fov_rad = np.deg2rad(fov_deg)

    cam_x = c2w[0, -1]
    cam_y = c2w[1, -1]
    cam_z = c2w[2, -1]

    corn1 = [np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0), -1.0]
    corn2 = [-np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0), -1.0]
    corn3 = [-np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0), -1.0]
    corn4 = [np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0), -1.0]

    corn1 = np.dot(c2w[:3, :3], corn1)
    corn2 = np.dot(c2w[:3, :3], corn2)
    corn3 = np.dot(c2w[:3, :3], corn3)
    corn4 = np.dot(c2w[:3, :3], corn4)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2) * zoom
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2) * zoom
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2) * zoom
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1] 
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2) * zoom
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]


    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T


class CameraVisualizer:

    def __init__(self, poses, legends, colors, images=None, mesh_path=None, pc_path=None, camera_x=1.0):
        self._fig = None

        self._camera_x = camera_x
        
        self._poses = poses
        self._legends = legends
        self._colors = colors

        self._raw_images = None
        self._bit_images = None
        self._image_colorscale = None
        
        if images is not None:
            self._raw_images = images
            self._bit_images = []
            self._image_colorscale = []
            for img in images:
                if img is None:
                    self._bit_images.append(None)
                    self._image_colorscale.append(None)
                    continue

                bit_img, colorscale = self.encode_image(img)
                self._bit_images.append(bit_img)
                self._image_colorscale.append(colorscale)

        self._mesh = None
        if mesh_path is not None and os.path.exists(mesh_path):
            import trimesh
            self._mesh = trimesh.load(mesh_path, force='mesh')
        self._pc = None
        if pc_path is not None and os.path.exists(pc_path):
            self._pc = np.load(pc_path)


    def encode_image(self, raw_image):
        '''
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        '''
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]
        
        return bit_image, colorscale


    def update_figure(
        self, 
        scene_bounds, 
        height=720,
        line_width=10,
        base_radius=0.0, 
        zoom_scale=1.0, 
        fov_deg=50., 
        mesh_z_shift=0.0, 
        mesh_scale=1.0, 
        show_background=False, 
        show_grid=False, 
        show_ticklabels=False, 
        y_up=False,
    ):

        fig = go.Figure()

        for i in range(len(self._poses)):
            pose = self._poses[i]
            clr = np.array([self._colors[i], self._colors[i]])
            legend = self._legends[i]

            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            if isinstance(fov_deg, float) or len(fov_deg) == 1:
                fov = fov_deg
            else:
                fov = fov_deg[i]
            cone = calc_cam_cone_pts_3d(pose, fov)
            radius = np.linalg.norm(pose[:3, -1])

            if self._bit_images and self._bit_images[i]:

                raw_image = self._raw_images[i]
                bit_image = self._bit_images[i]
                colorscale = self._image_colorscale[i]

                (H, W, C) = raw_image.shape

                z = np.zeros((H, W)) + base_radius
                scale = np.linalg.norm(cone[1] - cone[2]) / 2
                (x, y) = np.meshgrid(np.linspace(-scale, scale, W), np.linspace(scale, -scale, H) * H / W)
                
                xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

                rot_xyz = np.matmul(xyz, pose[:3, :3].T) + pose[:3, -1]

                offset = cone[2] - rot_xyz[0, 0, :]
                rot_xyz += offset.reshape((1, 1, 3))
                
                x, y, z = rot_xyz[:, :, 0], rot_xyz[:, :, 1], rot_xyz[:, :, 2]
                
                fig.add_trace(go.Surface(
                    x=x, y=y, z=z,
                    surfacecolor=bit_image,
                    cmin=0,
                    cmax=255,
                    colorscale=colorscale,
                    showscale=False,
                    lighting_diffuse=1.0,
                    lighting_ambient=1.0,
                    lighting_fresnel=1.0,
                    lighting_roughness=1.0,
                    # lighting_specular=0.3))
                    lighting_specular=0,
                    showlegend=False))
            
            for (j, edge) in enumerate(edges):
                (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                fig.add_trace(go.Scatter3d(
                    x=[x1, x2], 
                    y=[y1, y2], 
                    z=[z1, z2], 
                    mode='lines',
                    line=dict(color=clr, width=line_width), 
                    showlegend=False))
            
            # Add label.
            if cone[0, 2] < 0:
                fig.add_trace(go.Scatter3d(
                    x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] - 0.05], showlegend=False,
                    mode='text', text=legend, textposition='bottom center'))
            else:
                fig.add_trace(go.Scatter3d(
                    x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] + 0.05], showlegend=False,
                    mode='text', text=legend, textposition='top center'))

        # look at the center of scene
        fig.update_layout(
            height=height,
            autosize=True,
            hovermode=False,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            showlegend=True,
            legend=dict(
                yanchor='bottom',
                y=0.01,
                xanchor='right',
                x=0.99,
            ),
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                    center=dict(x=0.0, y=0.0, z=0.0),
                    up=dict(x=0.0, y=0.0, z=1.0)),
                xaxis_title='X',
                yaxis_title='Z' if y_up else 'Y',
                zaxis_title='Y' if y_up else 'Z',
                xaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                yaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                zaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks='')
            )
        )

        self._fig = fig
        return fig
