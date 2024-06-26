import torch
import ttnn


def add_pat(x, y):
    return torch.ops.aten.add.Tensor(x, y)


def add_rep(x, y):
    return ttnn.add(x, y)


def add_pat2(x):
    return torch.ops.aten.add.Tensor(x, x)


def add_rep2(x):
    return ttnn.add(x, x)


pat_rep_list = [
    (add_pat, add_rep),
    (add_pat2, add_rep2),
]
