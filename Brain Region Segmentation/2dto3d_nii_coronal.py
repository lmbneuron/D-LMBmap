import os
import numpy as np
import SimpleITK as sitk
import cv2
import config

if __name__ == '__main__':
    args = config.args
    root = 'output/' + args.name
    dirs = sorted(os.listdir(root))
    for d in dirs:
        if 'result' in d:
            d2_root = os.path.join(root, d)
            n = 'r1_' + args.name + '_M_cea_'
            d3_root = os.path.join(root, 'c',  n + d + '.nii')
            if not os.path.exists(os.path.join(root, 'c')): os.mkdir(os.path.join(root, 'c'))
            print(d2_root)
            images = sorted(os.listdir(d2_root))
            nii = []
            nii_new =[]
            _x, _y = 528, 456
            for image in images:
                if 'DS_Store' not in image:
                    img = sitk.ReadImage(os.path.join(d2_root, image))
                    img = sitk.GetArrayFromImage(img)
                    img = np.asarray(img)

                    nii.append(img)


            nii = np.array(nii)
            print(nii.shape)
            nii = nii.transpose(1, 2, 0)
            print(nii.shape)
                
            for img in nii:
                img1 = np.zeros_like(img)
                img1[img == 1] = 1
                img1_new = cv2.resize(img1, (_x, _y))
                
                img2 = np.zeros_like(img)
                img2[img == 2] = 1
                img2_new = cv2.resize(img2, (_x, _y))
                
                img_new = np.zeros_like(img1_new)
                img_new[img1_new == 1] = 1
                img_new[img2_new == 1] = 2

                nii_new.append(img_new)
                
            nii_new = np.array(nii_new)
            nii_new = nii_new.transpose(2, 0, 1)
            print(nii_new.shape)

            nii_file = sitk.GetImageFromArray(nii_new.astype(np.uint8))
            sitk.WriteImage(nii_file, d3_root)


