from plyfile import PlyData,PlyElement
import numpy as np
from ms_regnet.tools.io import read_mhd
from random import randint


def add_color(points, color=(255, 0, 0)):
    point_num = points.shape[0]
    points = [(points[i,0], points[i,1], points[i,2], color[0], color[1], color[2]) for i in range(point_num)]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'), ('red', 'u1'),
                                     ('green', 'u1'), ('blue', 'u1')])
    return vertex


def write_ply(save_path,points, color=(255, 0, 0), text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    el = PlyElement.describe(points, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)



path = r"C:\zht\backup\code\Recursive_network_pytorch\data\moving\210325_Adult_Brain_11_redo_13-41-38\test_result17_opt_py_hole.tiff"
color = (randint(0, 255), randint(0, 255), randint(0, 255))
image = read_mhd(path)
points = np.nonzero(image)
points = np.array(points)
points = points.T
color_points = add_color(points, color)

print(".".join(path.split(".")[:-1])+".ply")
write_ply(".".join(path.split(".")[:-1])+".ply", color_points)