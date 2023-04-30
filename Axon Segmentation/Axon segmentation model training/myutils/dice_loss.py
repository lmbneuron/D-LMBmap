import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

def iou(pre, label):
    eps = 1e-8
    if pre.sum() == 0:
        print('zero')
    return (pre * label).sum() / ((pre + label).sum() - (pre * label).sum() + eps)


def t_iou(pre, label):
    eps = 1e-8

    label_ori = label.clone()
    p1 = torch.nn.functional.max_pool3d(label, (3, 1, 1), 1, (1, 0, 0))
    p2 = torch.nn.functional.max_pool3d(label, (1, 3, 1), 1, (0, 1, 0))
    p3 = torch.nn.functional.max_pool3d(label, (1, 1, 3), 1, (0, 0, 1))
    label = torch.max(torch.max(p1, p2), p3) - label_ori

    return (pre * (label_ori + label)).sum() / ((pre * label + label_ori + pre - label_ori * pre).sum() + eps)

def junk_ratio(pre, label):
    difference = pre + label - 2 * pre * label
    difference = difference.view(pre.shape[0], -1)
    junks = label.view(label.shape[0], label.shape[1], -1).sum(axis=2)
    junks = (junks == 0).float()
    # axons = (junks != 0).float()
    # wrong_axons = (difference * axons).sum()
    wrong_junks = (difference * junks).sum()
    return wrong_junks / (difference.sum() + 1e-10)

def dice_error(input, target):
    smooth = 1.
    num = input.size(0)
    m1 = input.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    # return (2. * intersection) / (m1.sum() + m2.sum())

def soft_skeletonize(x, thresh_width=5):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        p1 = torch.nn.functional.max_pool3d(x * -1, (3, 1, 1), 1, (1, 0, 0)) * -1
        p2 = torch.nn.functional.max_pool3d(x * -1, (1, 3, 1), 1, (0, 1, 0)) * -1
        p3 = torch.nn.functional.max_pool3d(x * -1, (1, 1, 3), 1, (0, 0, 1)) * -1
        min_pool_x = torch.min(torch.min(p1, p2), p3)
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool3d(min_pool_x, (3, 3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

def positive_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    clf = center_line.view(*center_line.shape[:2], -1)
    vf = vessel.view(*vessel.shape[:2], -1)

    intersection = (clf * vf).sum(-1)
    return (intersection.sum(0) + 1e-12) / (clf.sum(-1).sum(0) + 1e-12)


def soft_cldice_f1(pred, target):
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice acc
    '''
    target_skeleton = soft_skeletonize(target)
    cl_pred = soft_skeletonize(pred)
    # save = cl_pred.cpu().numpy()[0][0] * 255
    # tifffile.imsave('/media/root/data4/szy/validate/155829/whole_label/1.tiff', save.astype(np.uint8))
    # tifffile.imsave('/media/root/data4/szy/validate/155829/whole_label/1.tiff',
    #                 target.cpu().numpy()[0][0].astype(np.uint8))
    clrecall = positive_intersection(target_skeleton, pred)  # ClRecall
    recall = positive_intersection(target, pred)
    clacc = positive_intersection(cl_pred, target)
    acc = positive_intersection(pred, target)
    return clrecall[0], clacc[0], recall[0], acc[0]
