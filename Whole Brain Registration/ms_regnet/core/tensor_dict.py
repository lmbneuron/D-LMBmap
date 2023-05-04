import torch


def tensor_cuda(data):
    dfs_cuda(data)
    return data


def tensor_cpu(data):
    dfs_cpu(data)
    return data


def dfs_cuda(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = data[k].cuda()
            else:
                dfs_cuda(data[k])


def dfs_cpu(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = data[k].detach().cpu()
            else:
                dfs_cpu(data[k])


def tensor_half(data):
    dfs_half(data)
    return data


def dfs_half(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = data[k].half()
            else:
                dfs_half(data[k])
