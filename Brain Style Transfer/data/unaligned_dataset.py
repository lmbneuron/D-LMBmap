import os
from data.base_dataset import BaseDataset, get_transform
import random
import numpy as np
import torch
import tifffile
import cv2

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dataroot = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.A_phase)
        self.dir_B = os.path.join(opt.dataroot, opt.B_phase)

        self.A_paths = sorted(self.load_file_name_list(self.dir_A))   # load images from '/train_data/trainA.txt'
        self.B_paths = sorted(self.load_file_name_list(self.dir_B))    # load images from '/train_data/trainB.txt'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # Load and normalize A and B data
        A_img = tifffile.imread(os.path.join(self.dataroot, A_path))
        new_x, new_y = 448, 320       #c
        A_img = cv2.resize(A_img, (new_x, new_y))
        dmin = A_img.min()
        dmax = A_img.max()
        A_img = (A_img - dmin) / (dmax - dmin + 1)
        A = A_img.astype(np.float32)
        B_img = tifffile.imread(os.path.join(self.dataroot, B_path))
        B_img = cv2.resize(B_img, (new_x, new_y))
        dmin = B_img.min()
        dmax = B_img.max()
        B_img = (B_img - dmin) / (dmax - dmin + 1)
        B = B_img.astype(np.float32)

        # apply image transformation
        A = self.transform_A.augment_images(A)
        B = self.transform_B.augment_images(B)

        # Save the data
        A_ = A * 255
        cv2.imwrite('temp/A.png', A_)
        B_ = B * 255
        cv2.imwrite('temp/B.png', B_)
        A = torch.from_numpy(A[np.newaxis, ...])
        B = torch.from_numpy(B[np.newaxis, ...])

        # Extract and save skull mask
        threshold = 5
        A_[A_ <= threshold] = 0
        A_[A_ > threshold] = 1
        cv2.imwrite('temp/real_A_mask.png', A_ * 255)
        A_mask = torch.from_numpy(A_.reshape(1, A_.shape[0], A_.shape[1]))
        B_[B_ <= threshold] = 0
        B_[B_ > threshold] = 1
        cv2.imwrite('temp/real_B_mask.png', B_ * 255)
        B_mask = torch.from_numpy(B_.reshape(1, B_.shape[0], B_.shape[1]))
        cv2.imwrite('temp/A1.png', A[0].numpy() * 255)
        cv2.imwrite('temp/B1.png', B[0].numpy() * 255)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_mask': A_mask, 'B_mask': B_mask}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()
                if not lines:
                    break
                    pass
                file_name_list.append(lines)
                pass
        return file_name_list
