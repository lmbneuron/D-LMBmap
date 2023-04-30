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


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class DiceLoss(Function):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, input, target, save=True):
        if save:
            self.save_for_backward(input, target)
        eps = 0.000001
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
#       print(input)
        intersect = torch.dot(result, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
            union, intersect, target_sum, result_sum, 2*IoU))
        out = torch.FloatTensor(1).fill_(2*IoU)
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_input , None

def dice_loss(input, target):
    return DiceLoss()(input, target)

def dice_error(input, target):
    smooth = 1.
    num = input.size(0)
    m1 = input.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
