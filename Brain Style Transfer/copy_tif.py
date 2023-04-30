import os
import tifffile
import numpy as np
from PIL import Image
import cv2

if __name__ == '__main__':
    root = 'results/maps_cyclegan/model40/test_latest/images'
    save = '/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/ljy/whole_2d/train_dataset'
    dirs = sorted(os.listdir(root))
    typ = '_fake'
    for d in dirs:
        d_new = d.replace('_m', '_r40_m')
        imgs = sorted(os.listdir(os.path.join(root, d)))
        if not os.path.exists(os.path.join(save, d_new)): os.mkdir(os.path.join(save, d_new))
        for i in range(len(imgs)):
            if 'fake_B' in imgs[i]:
                im = np.array(Image.open(os.path.join(root, d, imgs[i])))
                im_name = imgs[i].split('_')[0] + '.tif'
                print(os.path.join(save, d_new, im_name))
                tifffile.imwrite(os.path.join(save, d_new, im_name), im.astype(np.uint8))
                
                