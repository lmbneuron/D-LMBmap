import torch
from random import randint
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np


def visual_img(mode, name, fix_img, mov_img, reg_img_list, writer, epoch):
    img_list = [fix_img, mov_img, *reg_img_list]

    img_slice = torch.cat([
                            torch.cat([img[i, :, :].unsqueeze(0) for img in img_list], 2)
                            for i in range(0, fix_img.shape[0], 10)], 1).cpu()
    header = _create_header(img_slice.shape, name)
    writer.add_image(mode+"/img/x", torch.cat((header, img_slice), 1), epoch)

    img_slice = torch.cat([
        torch.cat([img[:, i, :].unsqueeze(0) for img in img_list], 2).cpu()
        for i in range(0, fix_img.shape[1], 10)], 1)
    header = _create_header(img_slice.shape, name)
    writer.add_image(mode+"/img/y", torch.cat((header, img_slice), 1), epoch)

    img_slice = torch.cat([
        torch.cat([img[:, :, i].unsqueeze(0) for img in img_list], 2).cpu()
        for i in range(0, fix_img.shape[2], 10)], 1)
    header = _create_header(img_slice.shape, name)
    writer.add_image(mode+"/img/z", torch.cat((header, img_slice), 1), epoch)


def _create_header(shape, name):
    header = np.zeros((100, shape[2], 1), dtype=np.uint8) + 255
    header = cv2.putText(header, name, (10, header.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)
    header = header.astype(np.float32) / 255
    header = np.transpose(header, (2, 0, 1))
    header = torch.Tensor(header)
    return header


def visual_deform_space(mode, dl, writer, epoch):
    index = randint(0, dl.shape[0]-1)
    writer.add_image(mode+"/space/x", dl[index, :, :].permute(2, 0, 1), epoch)

    index = randint(0, dl.shape[1]-1)
    writer.add_image(mode+"/space/y", dl[:, index, :].permute(2, 0, 1), epoch)

    index = randint(0, dl.shape[2]-1)
    writer.add_image(mode+"/space/z", dl[:, :, index].permute(2, 0, 1), epoch)

    writer.add_histogram(mode+"/dl/x", dl[..., 0], epoch)
    writer.add_histogram(mode+"/dl/y", dl[..., 1], epoch)
    writer.add_histogram(mode+"/dl/z", dl[..., 2], epoch)


def visual_gradient(model: Module, writer: SummaryWriter, epoch: int):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram("grad/"+name, param.grad, epoch)


def visual_delta_space(mode, dl, writer, epoch):
    writer.add_histogram(mode+"/delta/dl/x", dl[..., 0], epoch)
    writer.add_histogram(mode+"/delta/dl/y", dl[..., 1], epoch)
    writer.add_histogram(mode+"/delta/dl/z", dl[..., 2], epoch)
