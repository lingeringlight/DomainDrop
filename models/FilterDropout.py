import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def mask_selection(scores, percent, wrs_flag):
    # input: scores: BxN
    batch_size = scores.shape[0]
    num_neurons = scores.shape[1]
    drop_num = int(num_neurons * percent)

    if wrs_flag == 0:
        # according to scores
        threshold = torch.sort(scores, dim=1, descending=True)[0][:, drop_num]
        threshold_expand = threshold.view(batch_size, 1).expand(batch_size, num_neurons)
        mask_filters = torch.where(scores > threshold_expand, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
    else:
        # add random modules
        score_max = scores.max(dim=1, keepdim=True)[0]
        score_min = scores.min(dim=1, keepdim=True)[0]
        scores = (scores - score_min) / (score_max - score_min)
        
        r = torch.rand(scores.shape).cuda()  # BxC
        key = r.pow(1. / scores)
        threshold = torch.sort(key, dim=1, descending=True)[0][:, drop_num]
        threshold_expand = threshold.view(batch_size, 1).expand(batch_size, num_neurons)
        mask_filters = torch.where(key > threshold_expand, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())

    mask_filters = 1 - mask_filters  # BxN
    return mask_filters


def filter_dropout_channel(scores, percent, wrs_flag):
    # scores: BxCxHxW
    batch_size, channel_num, H, W = scores.shape[0], scores.shape[1], scores.shape[2], scores.shape[3]
    channel_scores = nn.AdaptiveAvgPool2d((1, 1))(scores).view(batch_size, channel_num)
    # channel_scores = channel_scores / channel_scores.sum(dim=1, keepdim=True)
    mask = mask_selection(channel_scores, percent, wrs_flag)   # BxC
    mask_filters = mask.view(batch_size, channel_num, 1, 1)
    return mask_filters


