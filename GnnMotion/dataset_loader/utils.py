import torch
import numpy as np


def one_hot(inp, label_count):
    out = torch.zeros(label_count, inp.numel(), dtype=torch.uint8, device=inp.device)
    out[inp.view(-1), torch.arange(out.shape[1])] = 1
    out = out.view((label_count,) + inp.shape)
    return out


