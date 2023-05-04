import numpy as np
import torch
import tqdm

from ms_regnet.constrain.deformer_zoo import DeformerZoo


class Inverter:
    def __init__(self, mode=0):
        self.mode = mode

    def invert(self, data, space):
        """
        :params data:
        :params space: torch's space data required
        """
        return self._invert_idm(data, space)


    def _invert_idm(self, data, space):
        """
        inverse the space by InverseDistanceWeighted.
        Inference: https://wenku.baidu.com/view/25a8f11480c4bb4cf7ec4afe04a1b0717fd5b37f.html?_wkts_=1669627209119&bdQuery=Inverse+Distance+Weighted
        """
        space = np.squeeze(space)
        shape = (space.shape[0], space.shape[1], space.shape[2])
        space, used_matrix = inv_space0(shape, space, return_used_matrix=True)
        ksize = 2
        space = space.astype(np.float32)
        new_space = space.copy()

        for i in tqdm.tqdm(range(ksize, space.shape[0]-ksize)):
            for j in range(ksize, space.shape[1]-ksize):
                for k in range(ksize, space.shape[2]-ksize):
                    if used_matrix[i, j, k]:
                        continue
                    cur_ksize = ksize
                    while True:
                        control_pts = np.array([(ii, jj, kk)
                                                for ii in range(max(0, i-cur_ksize), min(space.shape[0], i + cur_ksize + 1))
                                                for jj in range(max(0, j-cur_ksize), min(space.shape[1], j + cur_ksize + 1))
                                                for kk in range(max(0, k-cur_ksize), min(space.shape[2], k + cur_ksize + 1))
                                                if used_matrix[ii, jj, kk]], dtype=np.float32)
                        control_val = np.array([space[ii, jj, kk]
                                                for ii in range(max(0, i - cur_ksize), min(space.shape[0], i + cur_ksize + 1))
                                                for jj in range(max(0, j - cur_ksize), min(space.shape[1], j + cur_ksize + 1))
                                                for kk in range(max(0, k - cur_ksize), min(space.shape[2], k + cur_ksize + 1))
                                                if used_matrix[ii, jj, kk]], dtype=np.float32)
                        if len(control_pts) > 0:
                            break
                        cur_ksize += 1
                    new_space[i, j, k] = interpolate_by_idm(control_val, control_pts, np.array([i, j, k], dtype=np.float32))
        new_space = np.flip(new_space, -1)
        new_space = np.ascontiguousarray(new_space)
        new_space[..., 0] = new_space[..., 0] / (new_space.shape[2]-1) * 2 - 1
        new_space[..., 1] = new_space[..., 1] / (new_space.shape[1]-1) * 2 - 1
        new_space[..., 2] = new_space[..., 2] / (new_space.shape[0]-1) * 2 - 1
        new_space = new_space[np.newaxis, ...]

        from torch.nn.functional import interpolate
        new_space = torch.Tensor(new_space, device=[i for i in data.values()][0]["img"].device)
        new_space = new_space.permute(0, 4, 1, 2, 3)
        new_space = interpolate(new_space, scale_factor=2, mode='trilinear')
        new_space = new_space.permute(0, 2, 3, 4, 1)

        new_data = {}
        for mk in data.keys():
            new_data[mk] = DeformerZoo.get_deformer_by_constrain(mk)(None, data[mk], new_space)
        return new_data, new_space


def inv_space0(result_shape, deform_image, return_used_matrix=False):
    def limit(value, max):
        return value >= 0 and value < max

    used_matrix = np.zeros(shape=result_shape, dtype=np.bool)
    map = np.zeros(shape=(result_shape[0], result_shape[1], result_shape[2], 3),
                   dtype=np.int16) + 500
    to_x = (deform_image[:, :, :, 2] + 1) * result_shape[0] / 2
    to_y = (deform_image[:, :, :, 1] + 1) * result_shape[1] / 2
    to_z = (deform_image[:, :, :, 0] + 1) * result_shape[2] / 2
    for i in range(np.shape(deform_image)[0]):
        for j in range(np.shape(deform_image)[1]):
            for k in range(np.shape(deform_image)[2]):
                x = int(round(to_x[i, j, k]))
                y = int(round(to_y[i, j, k]))
                z = int(round(to_z[i, j, k]))
                if limit(x, result_shape[0]) and \
                        limit(y, result_shape[1]) and \
                        limit(z, result_shape[2]):
                    map[x][y][z] = (i, j, k)
                    used_matrix[x][y][z] = 1
    if return_used_matrix:
        return map, used_matrix
    else:
        return map


def interpolate_by_idm(sub_cpt_ref, tar_cpt_ref, center_pt):
    """
    given the (0-3, 0-3, 0-3) space and calculate the mid point's value
    :param sub_cpt_ref: which value does the control point map to in the moving image.
    :param tar_cpt_ref: control points in fix image, x, y, z range from 0-img_size.
    :param img_size: the cube size.
    """
    p = 2
    center_pt = center_pt[np.newaxis, :]
    dis = np.sqrt(np.sum((tar_cpt_ref - center_pt) ** 2, -1))

    dis = np.power(dis, -p)
    assert np.sum(dis) > 0
    dis /= np.sum(dis)
    return np.sum(dis[:, np.newaxis] * sub_cpt_ref, 0)
