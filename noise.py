
import numpy as np
import torch
import torch.nn.functional as F

def odd(w):
    return list(np.arange(1, w, step=2, dtype='long'))

def even(w):
    return list(np.arange(0, w, step=2, dtype='long'))

def white(x):
    fw, tw = x.shape[1], x.shape[2]

    first = F.relu(torch.normal(mean=torch.zeros(fw, tw), std=torch.ones(fw, tw)) ) * 0.05
    second_seed = F.relu(torch.normal(mean=torch.zeros(fw//2, tw//2), std=torch.ones(fw//2, tw//2))) * 0.03
    second = torch.zeros(fw, tw)

    row_x  = torch.zeros(int(fw//2), tw)
    row_x = torch.zeros(int(fw/2), tw)

    row_x[:, odd(tw)]  = second_seed
    row_x[:, even(tw)] = second_seed

    second[odd(fw), :]  = row_x
    second[even(fw), :] = row_x

    return second + first