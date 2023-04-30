import os
import numpy as np
import SimpleITK as sitk
import cv2
import config


if __name__ == '__main__':
    args = config.args
    root = 'output/' + args.name
    dirs = sorted(os.listdir(root))
    print(root)
    for d in dirs:
        if 'result' in d:
            d2_root = os.path.join(root, d)
            n = 'r1_' + args.name + '_M_cea_'
            d3_root = os.path.join(root,'a', n + d + '.nii')
            if not os.path.exists(os.path.join(root, 'a')): os.mkdir(os.path.join(root, 'a'))
            print(d2_root)
            images = sorted(os.listdir(d2_root))
            nii = []

            #a
            nii_new = np.zeros((320, 456, 528))
            _x, _y = 528, 456
            for image in images:
                if 'DS_Store' not in image:
                    img = sitk.ReadImage(os.path.join(d2_root, image))
                    img = sitk.GetArrayFromImage(img)
                    img = np.asarray(img)

                    img1 = np.zeros_like(img)
                    img1[img == 1] = 1
                    img1_new = cv2.resize(img1, (_x, _y))

                    img2 = np.zeros_like(img)
                    img2[img == 2] = 1
                    img2_new = cv2.resize(img2, (_x, _y))


                    img_new = np.zeros_like(img1_new)
                    img_new[img1_new == 1] = 1
                    img_new[img2_new == 1] = 2

                    nii.append(img_new)

            nii = np.array(nii)
            print(nii.shape)
            left = int(images[0].split('.')[0])
            right = int(images[-1].split('.')[0]) + 1
            print(left, right)
            nii_new[left: right] = nii
            print(nii_new.shape)
            nii_file = sitk.GetImageFromArray(nii_new.astype(np.uint8))
            sitk.WriteImage(nii_file, d3_root)


