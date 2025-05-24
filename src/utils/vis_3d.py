import torch
import numpy as np

"""
color reference: https://colorbrewer2.org/
"""

color_map = [
    [251,180,174],
    [179,205,227],
    [204,235,197],
    [222,203,228],
    [254,217,166],
    [255,255,204],
]


def vis_points(points, colors=None, labels=None, color_map='Paired', opacity=1.0, point_size=5.0):
    """Visualize a point cloud
    Note about direction in the visualization:  x: horizontal right (red arrow), y: vertical up (green arrow), and z: inside (blue arrow)
    Args:
        points ([np.array]): [N, 3] numpy array 
        colors ([type], optional): [description]. Defaults to None.
    """
    import pyvista as pv
    import numpy as np
    from pyvista import themes
    my_theme = themes.DefaultTheme()
    my_theme.color = 'black'
    my_theme.lighting = True
    my_theme.show_edges = True
    my_theme.edge_color = 'white'
    my_theme.background = 'white'
    pv.set_plot_theme(my_theme)

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()

    if colors is None and labels is not None:
        from matplotlib import cm
        if isinstance(colors, torch.Tensor):
            labels = labels.cpu().numpy()
        color_maps = cm.get_cmap(color_map, labels.max() + 1)
        colors = color_maps(labels)
    plotter = pv.Plotter()
    plotter.add_points(points, opacity=opacity, point_size=point_size, render_points_as_spheres=True, scalars=colors, rgb=True)
    plotter.show()


# show multiple point clouds at once in splitted windows. 
def vis_multi_points(points, colors=None, labels=None, 
                     opacity=1.0, point_size=5.0,
                     color_map='Paired', save_fig=False, save_name='example'):
    """Visualize a point cloud

    Args:
        points (list): a list of 2D numpy array. 
        colors (list, optional): [description]. Defaults to None.
    
    Example:
        vis_multi_points([points, pts], labels=[self.sub_clouds_points_labels[cloud_ind], labels])
    """
    import pyvista as pv
    import numpy as np
    from pyvista import themes
    from matplotlib import cm

    my_theme = themes.ParaViewTheme()
    my_theme.color = 'black'
    my_theme.lighting = True
    my_theme.show_edges = True
    my_theme.edge_color = 'white'
    my_theme.background = 'white'
    pv.set_plot_theme(my_theme)

    n_clouds = len(points)
    plotter = pv.Plotter(shape=(1, n_clouds), border=False)

    if colors is None:
        colors = [None] * n_clouds
    if labels is None:
        labels = [None] * n_clouds

    for i in range(n_clouds):
        plotter.subplot(0, i)
        if len(points[i].shape) == 3: points[i] = points[i][0]
        if colors[i] is not None and len(colors[i].shape) == 3: colors[i] = colors[i][0]
        if colors[i] is None and labels[i] is not None:
            color_maps = cm.get_cmap(color_map, labels[i].max() + 1)
            colors[i] = color_maps(labels[i])[:, :3]
            if colors[i].min() <0:
                colors[i] = np.array((colors[i] - colors[i].min) / (colors[i].max() - colors[i].min()) *255).astype(np.int8)
                
        plotter.add_points(points[i], opacity=opacity, point_size=point_size, render_points_as_spheres=True, scalars=colors[i], rgb=True)
    plotter.link_views()
    if save_fig:
        # plotter.show(auto_close=False)
        # plotter.screenshot(filename=f'{save_name}.png')
        plotter.show(screenshot='airplane.png')
        plotter.close()
    else:
        plotter.show()
        plotter.close()
    

def vis_neighbors(points, neighbor_points, point_index, 
                  colors='black', neighbor_colors='red', 
                  opacity=0.1, point_size=3.0, neighor_point_size=10):
    import pyvista as pv
    import numpy as np
    from pyvista import themes
    my_theme = themes.DefaultTheme()
    my_theme.color = 'black'
    my_theme.lighting = True
    my_theme.show_edges = True
    my_theme.edge_color = 'white'
    my_theme.background = 'white'
    pv.set_plot_theme(my_theme)

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
    if isinstance(neighbor_points, torch.Tensor):
        neighbor_points = neighbor_points.cpu().numpy()

    plotter = pv.Plotter()
    plotter.add_points(points, opacity=0.5, point_size=point_size, render_points_as_spheres=True)
    plotter.add_points(neighbor_points[point_index, :, :], point_size=neighor_point_size, color=neighbor_colors, render_points_as_spheres=True)
    plotter.add_points(points[point_index], point_size=neighor_point_size*2, color='green', render_points_as_spheres=True)
    plotter.show()


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %f %f %f\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()

"""    
def write_segmented_obj(data, out_filename):
    points = data['pos']
    cls = data['cls']
    seg = data['y']
    N = points.shape[0]
    colors = color_map[cls]
    print(colors)
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %f %f %f\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()
"""

def read_obj(filename):
    values = np.loadtxt(filename, usecols=(1,2,3,4,5,6))
    return values[:, :3], values[:, 3:6]