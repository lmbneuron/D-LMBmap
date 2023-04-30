import random
import tifffile
import itertools
import torch.utils.data as data

from tqdm import tqdm
from skimage import exposure
from utils import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from skimage.exposure import match_histograms


class AugAxonsDataset(data.Dataset):
    def __init__(self, opt):
        data_path = opt.dataroot
        n_samples = opt.n_samples
        input_dim = opt.input_dim
        data_mix = not opt.noartifacts
        self.opt = opt
        self.augment = dataAugmentation()
        self.unlabel = opt.semi
        self.flag = opt.isTrain

        self.datas = []
        self.labels = []
        self.labels_ori = []

        data_path = join(data_path, 'train') if self.flag else join(data_path, 'val')
        self.data_path = data_path
        volumes_folder_path = join(data_path, "volumes")
        labels_folder_path = join(data_path, "labels_sk") if self.flag else join(data_path, 'labels')

        volumes_path = get_dir(volumes_folder_path)
        labels_path = get_dir(labels_folder_path)

        assert len(labels_path) == len(volumes_path)
        if n_samples == None:
            n_samples = len(labels_path)

        total_volumes = 0
        with tqdm(total=len(volumes_path), desc=f'volumes numbers') as pbar:
            for vpath, lpath in zip(volumes_path, labels_path):
                # assert (vpath.split('/')[-1].replace('volume', 'label')) == lpath.split('/')[-1]
                volume = read_tiff_stack(vpath)
                label = read_tiff_stack(lpath)
                if volume.shape[0] < opt.input_dim or volume.shape[1] < opt.input_dim \
                        or volume.shape[2] < opt.input_dim:
                    continue
                # volume = (volume - volume.min()) / (volume.max() - volume.min())
                self.datas.append(volume)
                self.labels.append(label)

                total_volumes += 1
                pbar.update()
        self.axon_nums = total_volumes
        if data_mix:
            artifacts_folder_path = data_path + '/artifacts/'
            artifacts_path = get_dir(artifacts_folder_path)
            with tqdm(total=len(artifacts_path), desc=f'artifacts numbers') as pbar:
                for apath in artifacts_path:
                    artifact = read_tiff_stack(apath)
                    if artifact.shape[0] < opt.input_dim or artifact.shape[1] < opt.input_dim \
                            or artifact.shape[2] < opt.input_dim:
                        print(artifact.shape)
                        continue

                    # artifact = (artifact - artifact.min()) / (artifact.max() - artifact.min())
                    self.datas.append(artifact)
                    self.labels.append(np.zeros_like(label))

                    total_volumes += 1
                    pbar.update()

        self.labeled_num = total_volumes
        if self.unlabel and self.flag:
            nonlabel_path = get_dir(data_path + '/nonlabel')
            with tqdm(total=len(nonlabel_path), desc=f'volumes for semi-supervised learning') as pbar:
                for vpath in nonlabel_path:
                    volume = read_tiff_stack(vpath)
                    if volume.shape[0] < opt.input_dim or volume.shape[1] < opt.input_dim \
                            or volume.shape[2] < opt.input_dim:
                        continue
                    self.datas.append(volume)
                    self.labels.append(np.zeros_like(label))

                    total_volumes += 1
                    pbar.update()

    def __getitem__(self, index):
        volumes = self.datas[index]
        labels = self.labels[index]

        # ------------------------------------------------------------------ #
        # cutmix
        if index < self.axon_nums:
            artifact = self.datas[random.randint(self.axon_nums, len(self) - 1)]
            z = random.randint(0, labels.shape[0])
            x = random.randint(0, labels.shape[1])
            y = random.randint(0, labels.shape[2])
            artifact_chunk = artifact[:z, :x, :y].copy()
            volumes[:z, :x, :y] = artifact_chunk
            labels[:z, :x, :y] = np.zeros_like(artifact_chunk)

        data, label = self.augment.data_augmentation(volumes, labels)
        # ------------------------------------------------------------------ #
        # histograms match
        # if index < self.axon_nums and self.opt.isTrain:
        #     reference = self.datas[random.randint(0, self.axon_nums-1)]
        #     data = match_histograms(data, reference)
        # elif index >= self.axon_nums and self.opt.isTrain:
        #     reference = self.datas[random.randint(self.axon_nums, len(self)-1)]
        #     data = match_histograms(data, reference)
        # ------------------------------------------------------------------ #
        data = data[np.newaxis, ...].astype(np.float32)
        data = data / 6553 * random.randint(20, 150) / 100
        # data = (data - data.min()) / (data.max() - data.min())
        if self.opt.output_nc != 1:
            label = label.astype(np.long)
        else:
            label = label[np.newaxis, ...].astype(np.float32)
        return data, label

    def __len__(self):
        return len(self.datas)
