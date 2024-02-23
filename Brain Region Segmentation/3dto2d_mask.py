"""3dto2d_mask.py is a tool that can convert a 3D label with the suffix .nii(or .tiff) into a 2D image with suffix .ome.nii。

The directory structure of the file is as follows:
    ├──mask_3d
    │  ├──MASK_NAME.nii
    │  ├──...
    ├──data (Automatic creation)
    │  ├──train_dataset_view1
    │  │  ├──MASK_NAME: in view1
    │  │  │   └──0000.ome.nii
    │  │  │   └──...
    │  ├──train_dataset_view2
    │  │  ├──MASK_NAME: in view2
    │  │  │   └──0000.ome.nii
    │  │  │   └──...
    └──3dto2d_mask.py

This code sets optional arguments. you can check the relevant instructions by executing the following command:

>python 3dto2d_mask.py -h
usage: 3dto2d_mask.py [-h] [--file_dir FILE_DIR] [--file_3d FILE_3D] [--save_dir SAVE_DIR] [--file_2d FILE_2D]
                      [--type TYPE] [--image_shape IMAGE_SHAPE]

optional arguments:
  -h, --help            show this help message and exit
  --file_dir FILE_DIR
  --file_3d FILE_3D
  --save_dir SAVE_DIR
  --file_2d FILE_2D
  --type TYPE
  --image_shape IMAGE_SHAPE

if you want to change arguments, you can refer to the following demo:

>python 3dto2d_mask.py --file_dir .\ --file_3d mask_3d --save_dir .\ --file_2d data --type CH --image_shape 512,320,488

"""

import os
import argparse

import SimpleITK as sitk
import numpy as np
import cv2


def read_file(file_path):
    imgs = sitk.ReadImage(file_path)
    imgs = sitk.GetArrayFromImage(imgs)
    # print(imgs.shape)
    return imgs


def save_file(img, save_path, i):
    save_name = os.path.join(save_path, str(i).zfill(4) + '.ome.nii')
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, save_name)


def convert(imgs, image_shape, save_path, view_type=1):
    nii_new = []
    [d1, d2, d3] = [int(i) for i in image_shape.split(',')]  # 448, 320

    for i in range(imgs.shape[view_type - 1]):
        if view_type == 2:
            img = imgs[:, i, :]
            resize = (d1, d2)  # 512 448
        elif view_type == 3:
            img = imgs[:, :, i]
            resize = (d1, d3)  # 512 320
        else:  # view_type == 1
            img = imgs[i, :, :]
            resize = (d3, d2)  # 448, 320

        img = cv2.resize(img, resize)
        img = img.astype(np.uint16)

        save_file(img, save_path, i)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, default='./', dest='file_dir')
    parser.add_argument('--file_3d', type=str, default='mask_3d', dest='file_3d')
    parser.add_argument('--save_dir', type=str, default='./', dest='save_dir')
    parser.add_argument('--file_2d', type=str, default='data', dest='file_2d')
    parser.add_argument('--type', type=str, default='CH', dest='type')
    parser.add_argument('--image_shape', type=str, default='512,320,448', dest='image_shape')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    image_shape = args.image_shape
    file_dir = os.path.join(args.file_dir, args.file_3d)
    save_dir = os.path.join(args.save_dir, args.file_2d)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_list = os.listdir(file_dir)
    for file_name in file_list:
        file_path = os.path.join(file_dir, file_name)
        save_view1 = os.path.join(save_dir, 'train_dataset_view1',
                                  'mask_' + args.type + '_' + os.path.splitext(file_name)[0])
        save_view2 = os.path.join(save_dir, 'train_dataset_view2',
                                  'mask_' + args.type + '_' + os.path.splitext(file_name)[0])
        if not os.path.exists(save_view1):
            os.makedirs(save_view1)
        if not os.path.exists(save_view2):
            os.makedirs(save_view2)

        imgs = read_file(file_path)
        convert(imgs, image_shape, save_view1, 1)  # view1
        convert(imgs, image_shape, save_view2, 2)  # view2

        print(file_name, ' Conversion successful!')
