import torch
import torch.nn as nn
from .FilterDropout import filter_dropout_channel


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None


def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)


class LayerDiscriminator(nn.Module):
    def __init__(self, num_channels, num_classes, grl=True, reverse=True, lambd=0.0, wrs_flag=1):
        super(LayerDiscriminator, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = nn.Linear(num_channels, num_classes)
        self.softmax = nn.Softmax(0)
        self.num_channels = num_channels

        self.grl = grl
        self.reverse = reverse
        self.lambd = lambd

        self.wrs_flag = wrs_flag

    def scores_dropout(self, scores, percent):
        mask_filters = filter_dropout_channel(scores=scores, percent=percent, wrs_flag=self.wrs_flag)
        mask_filters = mask_filters.cuda()  # BxCx1x1
        return mask_filters

    def norm_scores(self, scores):
        score_max = scores.max(dim=1, keepdim=True)[0]
        score_min = scores.min(dim=1, keepdim=True)[0]
        scores_norm = (scores - score_min) / (score_max - score_min)
        return scores_norm

    def get_scores(self, feature, labels, percent=0.33):
        weights = self.model.weight.clone().detach()  # num_domains x C
        domain_num, channel_num = weights.shape[0], weights.shape[1]
        batch_size, _, H, W = feature.shape[0], feature.shape[1], feature.shape[2], feature.shape[3]

        weight = weights[labels].view(batch_size, channel_num, 1).expand(batch_size, channel_num, H * W)\
            .view(batch_size, channel_num, H, W)

        right_score = torch.mul(feature, weight)
        right_score = self.norm_scores(right_score)

        # right_score_masks: BxCxHxW
        right_score_masks = self.scores_dropout(right_score, percent=percent)
        return right_score_masks

    def forward(self, x, labels, percent=0.33):
        if self.grl:
            x = grad_reverse(x, self.lambd, self.reverse)

        feature = x.clone().detach()  # BxCxHxW
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # BxC
        y = self.model(x)

        # This step is to compute the 0-1 mask, which indicate the location of the domain-related information.
        # mask_filters: {0 / 1} BxCxHxW
        mask_filters = self.get_scores(feature, labels, percent=percent)
        return y, mask_filters
