import torch

__all__ = ["fusion_loss"]


def fusion_loss(output: dict, loss_cfg, constrain_dict: dict):
    '''
    :param output: loss dict gain from network
    :param loss_cfg: loss config
    :param constrain_dict: which constrains are used
    :return:
    '''
    loss = 0
    if loss_cfg["factor"].get("use_factor", True):
        for name, val in loss_cfg["factor"].items():
            if output.get(name) is not None:
                loss += val * output[name]

    for k, v in constrain_dict.items():
        if v:
            loss += output.get(k, 0) * loss_cfg["constrain"].get(k, 1)
            torch.cuda.empty_cache()
    return loss

    
if __name__ == "__main__":
    from ..loss import MILoss
    net = MILoss()
    for _ in range(10):
        i = torch.rand((2, 1, 80, 114, 132))
        j = torch.rand((2, 1, 80, 114, 132))
        i.requires_grad = True
        j.requires_grad = True
        i = i.cuda()
        j = j.cuda()
        loss = net(i, j)
        print(loss)
        loss.backward()
        for name, param in net.named_parameters():
            print(name, param)

