from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from loss import ContrastiveLoss
from layers import EncoderImageRegions, EncoderTextBert, CrossGraphFusionLayer

logger = logging.getLogger(__name__)


class BGMA(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImageRegions(opt)
        self.txt_enc = EncoderTextBert(opt)
        self.fusion = CrossGraphFusionLayer(opt)
        self.criterion = ContrastiveLoss(margin=opt.margin, max_violation=opt.max_violation)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.fusion.cuda()
            cudnn.benchmark = True

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.fusion.parameters())

        self.params = params
        all__params = list(self.txt_enc.parameters())
        bert_params = list(self.txt_enc.bert.parameters())
        bert_params_ptr = [p.data_ptr() for p in bert_params]
        params_no_bert = list()
        for p in all__params:
            if p.data_ptr() not in bert_params_ptr:
                params_no_bert.append(p)
        self.optimizer = torch.optim.Adam([
            {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
            {'params': params_no_bert, 'lr': opt.learning_rate},
            {'params': bert_params, 'lr': opt.learning_rate*0.02},
            {'params': self.fusion.parameters(), 'lr': opt.learning_rate}
        ],
            lr=opt.learning_rate)

        self.Eiters = 0
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.fusion.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.fusion.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.fusion.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.fusion.eval()

    def forward_emb(self, images, captions):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions)
        return img_emb, cap_emb

    def forward_scores(self, img_emb, cap_emb, lengths):
        scores = self.fusion(img_emb, cap_emb, lengths)
        return scores

    def forward_loss(self, scores, **kwargs):
        loss = self.criterion(scores)
        return loss

    def train_emb(self, images, captions, lengths, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions)
        scores = self.forward_scores(img_emb, cap_emb, lengths)
        loss = self.forward_loss(scores[0])
        loss_i2t = self.forward_loss(scores[1])
        loss_t2i = self.forward_loss(scores[2])
        self.optimizer.zero_grad()
        self.logger.update('Le', loss.item(), img_emb.size(0))
        self.logger.update('Le_i2t', loss_i2t.item(), img_emb.size(0))
        self.logger.update('Le_t2i', loss_t2i.item(), img_emb.size(0))

        loss = loss + 0.6*loss_i2t + 0.4*loss_t2i
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()


