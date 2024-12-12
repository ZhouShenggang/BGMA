import torch
import math
import numpy as np
import random
import os


def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(im, s):
    return im.mm(s.t())


def cosine_sim2(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def order_sim(im, s):
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1)) -
           im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def matrix_sim(im, s, len):
    batch_size = im.size(0)
    scores = []
    for i in range(s.size(0)):
        cap_i = s[i][:len[i], :].unsqueeze(0).contiguous()
        cap_i = cap_i.repeat(batch_size, 1, 1)
        sim_matrix = torch.bmm(cap_i, torch.transpose(im, 1, 2))  # B,nword, nregion
        sim = sim_matrix.max(2)[0].sum(1).unsqueeze(1)
        scores.append(sim)
    scores = torch.cat(scores, dim=1)
    return scores


def gelu(x):
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

