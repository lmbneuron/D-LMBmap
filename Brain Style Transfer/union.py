import os
import tifffile
import numpy as np
from PIL import Image
import cv2

if __name__ == '__main__':
    root = 'results/model1/test_latest/images'
    dirs = sorted(os.listdir(root))
    typ = '_fake'
    for d in dirs:
        imgs = sorted(os.listdir(os.path.join(root, d)))
        print(len(imgs))
        tiff = []
        
        for i in range(len(imgs)):
            if 'fake_B' in imgs[i]:
                im = np.array(Image.open(os.path.join(root, d, imgs[i])))
                im = cv2.resize(im, (456, 320), interpolation=cv2.INTER_CUBIC)
                print(os.path.join(root, d, imgs[i]))
                print(im.shape)
                tiff.append(im)
        
        tiff = np.array(tiff)
        print(tiff.shape)
        img = tiff
        tifffile.imwrite(os.path.join(root, d + typ + '.tiff'), img.astype(np.uint8))

