from __future__ import division
from __future__ import print_function

import logging
import sys
import time
import torch
import numpy as np
from collections import OrderedDict
import data_bert
from model import BGMA

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


@torch.no_grad()
def encode_data(model, data_loader):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None
    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        model.logger = val_logger
        img_emb, cap_emb = model.forward_emb(images, captions)

        if img_embs is None:
            if len(img_emb.shape) == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
                cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            cap_lens = [0] * len(data_loader.dataset)

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        if len(cap_emb.shape) != 3:
            cap_embs[ids, :] = cap_emb.data.cpu().numpy().copy()
        else:
            cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()
        for j, nid in enumerate(ids):
            cap_lens[nid] = lengths[j]

        del images, captions

    return img_embs, cap_embs, cap_lens


def shard_xattn(model, images, captions, cap_lens, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images) - 1) // shard_size + 1  # 5000/128 +1 = 40
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            with torch.no_grad():
                img = torch.from_numpy(images[im_start:im_end]).cuda().float()
                txt = torch.from_numpy(captions[cap_start:cap_end]).cuda().float()
                length = cap_lens[cap_start:cap_end]
            sim = model.forward_scores(img, txt, length)
            d[im_start:im_end, cap_start:cap_end] = sim[0].data.cpu().numpy()
    sys.stdout.write('\n')
    return d


@torch.no_grad()
def evalrank(model_path, fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.workers = 8
    print('Testing the best model')
    print(opt)
    print(checkpoint['epoch'])

    # construct model
    model = BGMA(opt)

    # model.make_data_parallel()
    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    print('Loading dataset')
    data_loader = data_bert.get_test_loader(opt)
    # val_loader = data_bert.get_val_loader(opt)

    logger.info('Computing results...')
    with torch.no_grad():
        start = time.perf_counter()
        img_embs, cap_embs, cap_lens = encode_data(model, data_loader)
        end = time.perf_counter()
        print("Encoding calculate time: {}".format(end - start))
    print('Images: %d, Captions: %d' % (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        start = time.perf_counter()
        sims = shard_xattn(model, img_embs, cap_embs, cap_lens, shard_size=100)
        npts = img_embs.shape[0]
        end = time.perf_counter()
        logger.info("calculate similarity time: {}".format(end - start))

        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            start = time.perf_counter()
            sims = shard_xattn(model, img_embs_shard, cap_embs_shard, cap_lens_shard, shard_size=100)
            npts = img_embs_shard.shape[0]
            end = time.perf_counter()
            logger.info("calculate similarity time: {}".format(end - start))

            r, rt0 = i2t(npts, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(npts, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[12]))
        print("Average i2t Recall: %.1f" % mean_metrics[10])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[11])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %mean_metrics[5:10])


def i2t(npts, sims, return_ranks=False, mode='coco'):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return r1, r5, r10, medr, meanr


def t2i(npts, sims, return_ranks=False, mode='coco'):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]

    if mode == 'coco':
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode == 'coco':
            for i in range(5):
                inds = np.argsort(sims[5 * index + i])[::-1]
                ranks[5 * index + i] = np.where(inds == index)[0][0]
                top1[5 * index + i] = inds[0]
        else:
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return r1, r5, r10, medr, meanr