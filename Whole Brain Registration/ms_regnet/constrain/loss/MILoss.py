import torch
from torch import nn
import torch.nn.functional as F
from math import pi, exp
from ..loss_zoo import LossZoo


@LossZoo.register(("simi", "mi"), ("tra", "mi"), ("cbx", "mi"))
class MILoss(nn.Module):
    def __init__(self, bin_num=40, dropout_number=200000):
        super(MILoss, self).__init__()
        self.dropout_number = dropout_number
        self.bin_num = bin_num
        self.k = 1.0 / 2
        self.sigma = 0.5 / self.k / self.bin_num  ##use 2*k*sigma=1/self.bin_num to calculate sigma
        self.min_number = exp(-self.k ** 2) / (2 * pi * self.sigma * self.sigma)
        self.mu1_list, self.sigma1_list = [], []
        self.mu2_list, self.sigma2_list = [], []
        for i in range(self.bin_num):
            mu1 = (2 * i + 1) / (2 * self.bin_num)
            for j in range(self.bin_num):
                mu2 = (2 * j + 1) / (2 * self.bin_num)
                self.mu1_list.append(mu1)
                self.mu2_list.append(mu2)
                self.sigma1_list.append(self.sigma)
                self.sigma2_list.append(self.sigma)

    def forward(self, fix: dict, mov: dict, reg: dict, deform_space: torch.Tensor):
        batch_x, batch_y = fix["img"], reg["img"]
        b = batch_x.shape[0]
        mu1 = torch.tensor(self.mu1_list, device=batch_x.device)
        mu1 = mu1.view(-1, 1, 1)
        mu2 = torch.tensor(self.mu2_list, device=batch_x.device)
        mu2 = mu2.view(-1, 1, 1)
        sigma1 = torch.tensor(self.sigma1_list, device=batch_x.device)
        sigma1 = sigma1.view(-1, 1, 1)
        sigma2 = torch.tensor(self.sigma2_list, device=batch_x.device)
        sigma2 = sigma2.view(-1, 1, 1)
        loss = 0
        for x, y in zip(batch_x, batch_y):
            x, y = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)
            if fix.get("ignore") is not None:
                ignore = fix["ignore"].bool()
                x, y = x[~ignore], y[~ignore]
            x, y = x.contiguous().view(1, 1, -1), y.contiguous().view(1, 1, -1)
            rand_index = torch.randint(0, x.shape[2] - 1, (self.dropout_number,))
            drop_out_x = x[:, :, rand_index]
            drop_out_y = y[:, :, rand_index]
            torch.cuda.empty_cache()
            '''codes below are refer to https://matthew-brett.github.io/teaching/mutual_information.html'''
            hgram = self._gauss(drop_out_x, drop_out_y, mu1, mu2, sigma1, sigma2)  # (bin_num**2, b, xyz)
            torch.cuda.empty_cache()
            hgram = hgram - self.min_number
            hgram = F.relu(hgram, inplace=True)
            hgram = torch.sum(hgram, -1)
            hgram = hgram * (2 * pi * self.sigma * self.sigma)
            torch.cuda.empty_cache()
            # print(hgram)
            pxy = hgram / torch.sum(hgram, 0, keepdim=True)
            pxy = pxy.view(self.bin_num, self.bin_num, 1)

            px = torch.sum(pxy, 1, keepdim=True)  # marginal for x over y
            py = torch.sum(pxy, 0, keepdim=True)  # marginal for y over x
            px_py = px * py  # Broadcast to multiply marginals
            # Now we can do the calculation using the pxy, px_py 2D arrays
            torch.cuda.empty_cache()
            loss += -torch.sum(pxy * torch.log(pxy + 1e-9) - pxy * torch.log(px_py + 1e-9))
        loss /= b
        return loss

    def _gauss(self, x, y, mu1, mu2, sigma1, sigma2):
        return 1.0 / (2 * pi * sigma1 * sigma2) * torch.exp(
            -0.5 * ((x - mu1) ** 2 / (sigma1 ** 2) + (y - mu2) ** 2 / (sigma2 ** 2)))


def mutual_information(hgram):
    """ Mutual information for joint histogram"""
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def cal(img1, img2, bin_num):
    for i in range(bin_num):
        for j in range(bin_num):
            count = 0
            l1, r1 = i / bin_num * 255, (i + 1) / bin_num * 255
            l2, r2 = j / bin_num * 255, (j + 1) / bin_num * 255
            for x in range(img1.shape[0]):
                for y in range(img1.shape[1]):
                    if img1[x, y] > l1 and img1[x, y] < r1 and img2[x, y] > l2 and img2[x, y] < r2:
                        count += 1
            print(count)


if __name__ == "__main__":
    # mi = MILoss()
    # x = torch.randn((2, 1, 32, 32, 32))
    # y = torch.randn((2, 1, 32, 32, 32))
    # out = mi(x, y)

    import numpy as np
    from ms_regnet.tools import read_tiff_stack

    bin_num = 15
    # img1 = cv2.imread(r"C:\Users\haimiao\Desktop\1.jpg", cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(r"C:\Users\haimiao\Desktop\2.jpg", cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.resize(img1, (128, 128))
    # img2 = cv2.resize(img2, (128, 128))
    # img2 = img1.astype(np.float) * 0.2
    # img2 = img2.astype(np.uint8)
    #
    # hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bin_num)
    # # print(hist_2d)
    # img1 = img1.astype(np.float)
    # img2 = img2.astype(np.float)
    # img1 /= 255
    # img2 /= 255
    # print(mutual_information(hist_2d))
    #
    # img1 = img1[np.newaxis, np.newaxis, ...]
    # img2 = img2[np.newaxis, np.newaxis, ...]

    img1 = read_tiff_stack(
        "/media/data1/zht/Recursive_network_pytorch/2021-04-19_12-39-48/210324_Adult_Brain_9_14-25-10/fix.tiff")
    img2 = read_tiff_stack(
        "/media/data1/zht/Recursive_network_pytorch/2021-04-19_12-39-48/210324_Adult_Brain_9_14-25-10/ela_reg_img.tiff")
    img3 = read_tiff_stack(
        "/media/data1/zht/Recursive_network_pytorch/2021-04-19_12-39-48/210324_Adult_Brain_9_14-25-10/reg.tiff")
    with torch.no_grad():
        mi = MILoss(bin_num)
        img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
        img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
        img3 = (img3 - np.min(img3)) / (np.max(img3) - np.min(img3))
        img1 = img1[np.newaxis, np.newaxis, ...]
        img2 = img2[np.newaxis, np.newaxis, ...]
        img3 = img3[np.newaxis, np.newaxis, ...]
        img1 = torch.tensor(img1, dtype=torch.float32)
        img2 = torch.tensor(img2, dtype=torch.float32)
        img3 = torch.tensor(img3, dtype=torch.float32)
        print(mi(img1, img2))
        print(mi(img1, img3))
        # print(mi(img1, img2))
