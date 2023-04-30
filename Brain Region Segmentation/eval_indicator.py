import SimpleITK as sitk
import os
import numpy as np
import config

def compute_dc_3d(pred, mask):
    inter = pred[pred == mask]
    inter_count = len(inter[inter == 1])

    pred_count = len(pred[pred == 1])
    mask_count = len(mask[mask == 1])

    dc = 2.0 * inter_count / (pred_count + mask_count)

    return dc

if __name__ == '__main__':
    args = config.args
    pred_root = 'output/' + args.name + '/c/'
    mask_root = 'output/mask/' + args.name + '/'

    preds = sorted(os.listdir(pred_root))
    masks = sorted(os.listdir(mask_root))
    print(preds, masks)
    s = 0    # Average
    for i in range(len(preds)):
        # Load the prediction
        pred = sitk.ReadImage(pred_root + preds[i])
        pred = sitk.GetArrayFromImage(pred)
        pred[pred == 1] = 0
        pred[pred == 2] = 1
        # Load the mask
        data_mask = sitk.ReadImage(mask_root + masks[i])
        data_mask = sitk.GetArrayFromImage(data_mask)
        data_mask1 = np.zeros_like(data_mask)
        data_mask1[data_mask == 1] = 0
        data_mask1[data_mask == 2] = 1

        dc1 = compute_dc_3d(pred, data_mask1)
        print("%.5f" % dc1)
        s += dc1

    s /= len(preds)
    print("%.5f" % s)
