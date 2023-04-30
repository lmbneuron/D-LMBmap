import torch
import torch.nn as nn
import torch.nn.functional as F


class PseudoLabelCrossEntropy(torch.nn.Module):
    def __init__(self, threshold=0.90, ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
    def forward(self, pred, target=None):
        # pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=True
        prob = torch.sigmoid(pred)
        backgroud = 1 - prob
        target = torch.ones_like(prob)
        target[prob >= self.threshold] = 1
        target[prob < self.threshold] = self.ignore_index

        ssl_loss = (-1 * prob * prob.log()).mean()
        return F.cross_entropy(torch.stack([prob, backgroud], dim=1), target.long(),
                               ignore_index=self.ignore_index) + ssl_loss


class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()

    def forward(self, p_logit):
        sg = nn.Sigmoid()
        p = sg(p_logit)
        out = -1 * p * (p + 1e-9).log()
        out = out.mean()
        return out