from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import random

import torch
import logging
import numpy as np
import tensorboard_logger as tb_logger
import argparse

import data_bert
from model import BGMA
from utils import seed_torch
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn, evalrank

seed_torch(100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/', help='path to datasets')
    parser.add_argument('--data_name', default='f30k', help='{coco, f30k}')
    parser.add_argument('--vocab_path', default='./vocab/', help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=25, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=50, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0002, type=float, help='Initial learning rate.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='./runs/', help='Path to save the model and Tensorboard ''log.')
    parser.add_argument('--model_name', default='./runs/', type=str)
    parser.add_argument('--resume', default='./runs/', type=str)
    parser.add_argument('--max_violation', action='store_false', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int, help='Dimensionality of the image embedding.')
    parser.add_argument('--measure', default='cosine', help='Similarity measure used (cosine|order)')
    parser.add_argument("--num_regions", type=int, default=36, help='max length of captions(containing <sos>,<eos>)')
    parser.add_argument('--seed', default=1, type=int)
    opt = parser.parse_args()

    postfix = datetime.now().strftime("%m%d_%H%M%S")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    opt.logger_name = opt.logger_name + opt.data_name + "/butd_region_bert_" + postfix + "/log"
    opt.model_name = opt.model_name + opt.data_name + "/butd_region_bert_" + postfix + "/checkpoints"

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    if not os.path.exists(opt.logger_name):
        os.makedirs(opt.logger_name)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load data loaders
    train_loader, val_loader = data_bert.get_loaders(opt)

    # Construct the model
    model = BGMA(opt)

    # optionally resume from a checkpoint
    start_epoch = 0
    best_rsum = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            best_r = checkpoint['best_r']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})".format(
                opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)
        # train for one epoch
        train(opt, train_loader, model, epoch)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model, epoch)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            },
            is_best,
            filename='checkpoint.pth',
            prefix=opt.model_name + '/')

    evalrank(opt.model_name + '/model_best.pth')


def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # if opt.reset_train:
        # Always reset to train mode, this is not the default behavior
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        images, captions, lengths, _ = train_data
        model.train_emb(images, captions, lengths)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader.dataset) // train_loader.batch_size + 1, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

            # Record logs in tensorboard
            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            tb_logger.log_value('step', i, step=model.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
            model.logger.tb_log(tb_logger, step=model.Eiters)


@torch.no_grad()
def validate(opt, val_loader, model, epoch):
    logger = logging.getLogger(__name__)
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens = encode_data(model, val_loader)
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    start = time.time()
    sims = shard_xattn(model, img_embs, cap_embs, cap_lens, shard_size=100)
    end = time.time()

    logger.info("calculate similarity time: {}".format(end - start))
    npts = img_embs.shape[0]
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanri) = t2i(npts, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanri))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r1i + r5i + r10 + r10i
    logger.info('Current rsum is {}'.format(currscore))

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    message = "Epoch: %d: Image to text: (%.1f, %.1f, %.1f) " % (epoch, r1, r5, r10)
    message += "Text to image: (%.1f, %.1f, %.1f) " % (r1i, r5i, r10i)
    message += "rsum: %.1f\n" % currscore

    log_file = os.path.join(opt.logger_name, "performance.log")
    logging_func(log_file, message)
    return currscore


def logging_func(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message)
    f.close()


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        torch.save(state, prefix + 'model_best.pth')


def adjust_learning_rate(opt, optimizer, epoch):
    logger = logging.getLogger(__name__)
    if epoch == 5:
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.4
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))
    if epoch == 10:
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.25
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))
    if epoch == 15:
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.5
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


if __name__ == '__main__':
    main()
