import torch
import numpy as np
from torch import nn
from transformers import BertModel
from utils import l2norm, cosine_sim2


class EncoderImageRegions(nn.Module):
    def __init__(self, opt):
        super(EncoderImageRegions, self).__init__()
        self.embed_size = opt.embed_size
        self.fc = nn.Linear(opt.img_dim, opt.embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)
        features = l2norm(features)
        return features


class EncoderTextBert(nn.Module):
    def __init__(self, opt):
        super(EncoderTextBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Sequential(nn.Linear(768, opt.embed_size), nn.ReLU(), nn.Dropout(0.1))

    def forward(self, x):
        cap = self.bert(x)
        cap_emb = l2norm(self.fc(cap[0]))
        return cap_emb


class CrossGraphFusionLayer(nn.Module):
    def __init__(self, opt):
        super(CrossGraphFusionLayer, self).__init__()
        self.opt = opt
        self.in_channels = 1024
        self.inter_channels = 1024

        conv_nd = nn.Conv1d
        self.out_1 = nn.utils.weight_norm(nn.Linear(16, 16))  # WN( fc(32,32) )
        self.out_2 = nn.utils.weight_norm(nn.Linear(16, 1))  # WN( fc(32,1) )
        self.weight_i2t = nn.Parameter(torch.ones((12, 1)))
        self.weight_t2i = nn.Parameter(torch.ones((36, 1)))

        self.i1 = conv_nd(in_channels=self.in_channels,
                          out_channels=self.inter_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0)
        nn.init.xavier_normal_(self.i1.weight)
        self.i2 = conv_nd(in_channels=self.in_channels,
                          out_channels=self.inter_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0)
        nn.init.xavier_normal_(self.i2.weight)
        self.t1 = conv_nd(in_channels=self.in_channels,
                          out_channels=self.inter_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0)
        nn.init.xavier_normal_(self.t1.weight)
        self.t2 = conv_nd(in_channels=self.in_channels,
                          out_channels=self.inter_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0)
        nn.init.xavier_normal_(self.t2.weight)

    def forward(self, img, cap, lengths):
        batch_size = img.size(0)
        scores = []
        scores_i2t = []
        scores_t2i = []

        v_img1 = self.i1(img.permute(0, 2, 1)).permute(0, 2, 1)  # [batch, 36, 1024]
        v_img2 = self.i2(img.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(len(cap)):
            cap_i = cap[i][:lengths[i], :].contiguous()
            cap_i = cap_i.repeat(batch_size, 1, 1)

            v_txt1 = self.t1(cap_i.permute(0, 2, 1)).permute(0, 2, 1)  # [batch, cap_len, 1024]
            v_txt2 = self.t2(cap_i.permute(0, 2, 1)).permute(0, 2, 1)

            attn_img1 = self.cal_attn(img, v_img1, 4)  # [batch_size, 36, 36]
            attn_i2t = self.cal_attn(img, v_txt1, 4)  # [batch, 36, cap_len]
            attn_txt1 = self.cal_attn(cap_i, v_txt1, 4)
            re_txt = torch.bmm(torch.bmm(torch.transpose(attn_i2t, 1, 2), attn_img1), attn_i2t)

            attn_txt2 = self.cal_attn(cap_i, v_txt2, 4)  # [batch_size, cap_len, cap_len]
            attn_t2i = self.cal_attn(cap_i, v_img2, 4)  # [batch, cap_pen, 36]
            attn_img2 = self.cal_attn(img, v_img2, 4)
            re_img = torch.bmm(torch.bmm(torch.transpose(attn_t2i, 1, 2), attn_txt2), attn_t2i)

            trans1 = torch.bmm(img.permute(0, 2, 1), attn_i2t).permute(0, 2, 1)  # cap_len, 1024
            trans2 = torch.bmm(cap_i.permute(0, 2, 1), attn_t2i).permute(0, 2, 1)  # 36, 1024

            sim_i2t = self.cal_mutilblock(cap_i, trans1)
            sim_t2i = self.cal_mutilblock(img, trans2)

            dis1_coarse = torch.cdist(re_txt.view(batch_size, 1, -1), attn_txt1.view(batch_size, 1, -1), p=2).view(
                batch_size, 1)
            dis1_fine = torch.cdist(re_txt.view(batch_size, re_txt.size(1), 1, -1),
                                    attn_txt1.view(batch_size, attn_txt1.size(1), 1, -1), p=2).view(batch_size, -1, 1)
            dis2_coarse = torch.cdist(re_img.view(batch_size, 1, -1), attn_img2.view(batch_size, 1, -1), p=2).view(
                batch_size, 1)
            dis2_fine = torch.cdist(re_img.view(batch_size, re_img.size(1), 1, -1),
                                    attn_img2.view(batch_size, attn_img2.size(1), 1, -1), p=2).view(batch_size, -1, 1)

            repeat_factor = cap_i.size(1) // self.weight_i2t.size(0) + 1
            weight_i2t = torch.cat([self.weight_i2t] * repeat_factor, dim=0)
            weight_i2t = weight_i2t[:cap_i.size(1)]
            sim = ((torch.mul(sim_i2t, weight_i2t) + 0.05 * torch.clamp(0.1 - dis1_fine, min=0)).view(batch_size, -1).mean(dim=1, keepdim=True) +
                   (torch.mul(sim_t2i, self.weight_t2i) + 0.05 *  torch.clamp(0.1 - dis2_fine, min=0)).view(batch_size, -1).mean(dim=1,
                                                                                                      keepdim=True))
            sim_i2t = sim_i2t.view(batch_size, -1).mean(dim=1, keepdim=True) + 0.01 *  torch.clamp(0.5 - dis1_coarse, min=0)
            sim_t2i = sim_t2i.view(batch_size, -1).mean(dim=1, keepdim=True) + 0.01 *  torch.clamp(0.5 - dis2_coarse, min=0)
            scores.append(sim)
            scores_i2t.append(sim_i2t)
            scores_t2i.append(sim_t2i)

        scores = torch.cat(scores, 1)
        scores_i2t = torch.cat(scores_i2t, 1)
        scores_t2i = torch.cat(scores_t2i, 1)

        return scores, scores_i2t, scores_t2i  # , dist_i2t, dist_t2i

    def cal_mutilblock(self, orig, fuse):
        qry_set = torch.split(orig, 64, dim=2)
        ctx_set = torch.split(fuse, 64, dim=2)

        qry_set = torch.stack(qry_set, dim=2)
        ctx_set = torch.stack(ctx_set, dim=2)
        node_vector = cosine_sim2(qry_set, ctx_set, dim=-1)  # vnode融合了文本

        sim = self.out_2(self.out_1(node_vector).tanh())

        return sim

    def cal_attn(self, Q, K, xlambda=1):
        batch_size, queryL = Q.size(0), Q.size(1)
        batch_size, sourceL = K.size(0), K.size(1)
        keyT = torch.transpose(K, 1, 2)
        attn = torch.bmm(Q, keyT)
        attn = attn.view(batch_size * queryL, sourceL)
        attn = nn.Softmax(dim=1)(attn * xlambda)
        attn = attn.view(batch_size, queryL, sourceL)

        return attn
