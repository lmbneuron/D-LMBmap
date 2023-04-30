from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import os
import tifffile

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dataroot = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.A_phase)

        self.A_paths = sorted(self.load_file_name_list(self.dir_A))
        # input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = tifffile.imread(os.path.join(self.dataroot, A_path))
        dmin = A_img.min()
        dmax = A_img.max()
        A_img = (A_img - dmin) / (dmax - dmin)
        A_img = A_img.astype(np.float32)
        tifffile.imsave('temp/s_a.tiff', A_img)
        A = torch.from_numpy(A_img[np.newaxis, ...])
#        print(A.shape)

        # A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                    pass
                file_name_list.append(lines)
                pass
        return file_name_list