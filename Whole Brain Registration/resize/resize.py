__all__= ('trilinear_interpolate', 'multi_channel_trilinear_inter')

import numpy as np
from PIL import Image
import numpy as np
EPS = 1e-9


def multi_channel_trilinear_inter(data: np.ndarray, dsize):
    """
    :param data: (x, y, z, c)
    :param dsize: (dx, dy, dz)
    :return: (dx, dy, dz, c)
    """
    data = np.transpose(data, (3, 0, 1, 2))
    new_data = []
    for x in data:
        x = trilinear_interpolate(x, dsize=dsize)
        new_data.append(x)
    new_data = np.array(new_data)
    new_data = np.transpose(new_data, (1, 2, 3, 0))
    return new_data


def trilinear_interpolate(data, dsize):
    assert data.ndim == 3, "data's ndim must be equal to 3"
    assert len(dsize) == 3, "dsize's ndim must be equal to 3"
    data = data.astype(np.float32)
    x_times, y_times, z_times = [dsize[i] / data.shape[i] for i in range(3)]
    data = np.pad(data, ((0, 1), (0, 1), (0, 1)), 'edge')

    print("now get new image x y z")
    new_all_points = np.arange(0, dsize[0] * dsize[1] * dsize[2], 1)
    new_all_points = new_all_points[:, np.newaxis]
    new_all_points = np.tile(new_all_points, (1, 3))
    new_all_points[:, 2] = new_all_points[:, 2] % dsize[2]
    new_all_points[:, 1] = (new_all_points[:, 1] % (dsize[2] * dsize[1])) // dsize[2]
    new_all_points[:, 0] = new_all_points[:, 0] // (dsize[2] * dsize[1])
    new_all_points = new_all_points.astype(np.float16)
    new_all_points = new_all_points / np.reshape([x_times, y_times, z_times], newshape=(1, 3))

    print("now start interpolate")
    '''now start interpolate'''

    '''x_0 y_0 z_0 is the smaller value of the grid point'''
    x_0, y_0, z_0 = np.floor(new_all_points[:, 0]), np.floor(new_all_points[:, 1]), np.floor(new_all_points[:, 2])

    '''x_1 y_1 z_1 is the bigger value of the grid point'''
    x_1, y_1, z_1 = np.ceil(new_all_points[:, 0]), np.ceil(new_all_points[:, 1]), np.ceil(new_all_points[:, 2])


    x_d = (new_all_points[:, 0] - x_0 + EPS) / (x_1 - x_0 + EPS)
    y_d = (new_all_points[:, 1] - y_0 + EPS) / (y_1 - y_0 + EPS)
    z_d = (new_all_points[:, 2] - z_0 + EPS) / (z_1 - z_0 + EPS)

    x_0, y_0, z_0 = x_0.astype(np.int16), y_0.astype(np.int16), z_0.astype(np.int16)
    x_1, y_1, z_1 = x_1.astype(np.int16), y_1.astype(np.int16), z_1.astype(np.int16)

    print("cal c")
    c_000 = data[x_0, y_0, z_0]
    c_100 = data[x_1, y_0, z_0]
    c_010 = data[x_0, y_1, z_0]
    c_001 = data[x_0, y_0, z_1]
    c_110 = data[x_1, y_1, z_0]
    c_101 = data[x_1, y_0, z_1]
    c_011 = data[x_0, y_1, z_1]
    c_111 = data[x_1, y_1, z_1]

    c = c_000 * (1 - x_d) * (1 - y_d) * (1 - z_d) \
        + c_100 * x_d * (1 - y_d) * (1 - z_d) \
        + c_010 * (1 - x_d) * y_d * (1 - z_d) \
        + c_001 * (1 - x_d) * (1 - y_d) * z_d \
        + c_110 * x_d * y_d * (1 - z_d) \
        + c_101 * x_d * (1 - y_d) * z_d \
        + c_011 * (1 - x_d) * y_d * z_d \
        + c_111 * x_d * y_d * z_d
    # print("c reshape")
    c = np.reshape(c, newshape=dsize)
    c = c.astype(np.float32)
    return c


def read_tiff_stack(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        try:
            img.seek(i)

            slice = np.array(img)
            images.append(slice)
        except Exception as e:
            print(e)
            continue

    return np.array(images)


def write_tiff_stack(vol, fname):
    im = Image.fromarray(vol[0])
    ims = []

    for i in range(1, vol.shape[0]):
        ims.append(Image.fromarray(vol[i]))

    im.save(fname, save_all=True, append_images=ims)
