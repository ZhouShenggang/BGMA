import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        # self.margin = margin
        self.margin = nn.Parameter(torch.tensor(margin), requires_grad=True)
        self.max_violation = max_violation

    def forward(self, score):
        diagonal = score.diag().view(score.shape[0], 1)
        d1 = diagonal.expand_as(score)
        d2 = diagonal.t().expand_as(score)

        cost_s = (self.margin + score - d1).clamp(min=0)
        cost_im = (self.margin + score - d2).clamp(min=0)

        mask = torch.eye(score.size(0)) > .5
        mask = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class ContrastiveLoss2(nn.Module):
    def __init__(self, margin=0.2, max_violation=False):
        super(ContrastiveLoss2, self).__init__()
        # self.margin = margin
        self.margin = nn.Parameter(torch.tensor(margin), requires_grad=True)
        self.max_violation = max_violation

    def forward(self, score):
        diagonal = score.diag().view(score.shape[0], 1)
        d1 = diagonal.expand_as(score)

        cost_s = (self.margin + score - d1).clamp(min=0)
        # cost_im = (self.margin + )
        mask = torch.eye(score.size(0)) > .5
        mask = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)

        return cost_s.sum()


class TripleContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, max_violation=False):
        super(TripleContrastiveLoss, self).__init__()
        # self.margin = margin
        self.contrastive1 = ContrastiveLoss(margin, max_violation)
        self.contrastive2 = ContrastiveLoss(margin, max_violation)
        self.contrastive3 = ContrastiveLoss(margin, max_violation)

    def forward(self, scores):
        loss1 = self.contrastive1(scores[0])
        loss2 = self.contrastive2(scores[1])
        loss3 = self.contrastive3(scores[2])
        loss = loss1 + 0.6 * loss2 + 0.4 * loss3
        return loss
