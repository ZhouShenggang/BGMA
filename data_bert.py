import torch
import torch.utils.data as data
import os
import numpy as np
from transformers import BertTokenizer


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split):
        loc = data_path + '/'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Captions
        self.data_split = data_split
        self.captions = []
        with open(loc + '%s_caps.txt' % data_split, 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
        self.length = len(self.captions)
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])

        caption = self.captions[index]
        caption_token = self.tokenizer.basic_tokenizer.tokenize(caption)
        output_tokens = []
        for i, token in enumerate(caption_token):
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                output_tokens.append(sub_token)
        output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
        targets = self.tokenizer.convert_tokens_to_ids(output_tokens)
        targets = torch.Tensor(targets)
        return image, targets, index

    def __len__(self):
        return self.length


def collate_fn(data):
    images, captions, index = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        targets[i, :lengths[i]] = torch.Tensor(cap[:lengths[i]])

    return images, targets, lengths, index


def get_precomp_loader(data_path, data_split, opt, shuffle=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(opt):
    dpath = os.path.join(opt.data_path, 'precomp/')

    train_loader = get_precomp_loader(dpath, 'dev', opt, True)
    val_loader = get_precomp_loader(dpath, 'dev', opt, False)

    return train_loader, val_loader


def get_val_loader(opt):
    dpath = os.path.join(opt.data_path, 'precomp/')
    val_loader = get_precomp_loader(dpath, 'dev', opt, False)
    return val_loader


def get_test_loader(opt):
    dpath = os.path.join(opt.data_path, 'precomp/')
    test_loader = get_precomp_loader(dpath, 'test', opt, False)

    return test_loader
