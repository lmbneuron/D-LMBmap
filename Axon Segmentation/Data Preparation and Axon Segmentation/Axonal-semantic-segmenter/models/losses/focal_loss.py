import torch
import torch.nn as nn


def FocalLoss(logit, target, gamma=2, alpha=0.5):
    n, c, h, w = logit.size()
    criterion = nn.CrossEntropyLoss().cuda()

    logpt = -criterion(logit, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt

    loss /= n

    return loss