import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, ignore_index, reduction='mean'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        '''
        params:
          - input (FloatTensor): (batch_size x n_classes)
          - target (LongTensor): (batch_size)

        return:
          - loss
        '''
        one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(-1), 1)
        eps = self.label_smoothing / (input.size(-1) - 1)
        weight = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * eps

        log_prob = F.log_softmax(input, dim=-1)     
        loss = -(weight * log_prob).sum(dim=-1)
        nll_loss = -(one_hot * log_prob).sum(dim=-1)

        non_pad_mask = target.ne(self.ignore_index)
        if self.reduction == "mean":
            loss = loss.masked_select(non_pad_mask).mean()
            nll_loss = nll_loss.masked_select(non_pad_mask).mean()
        elif self.reduction == "sum":
            loss = loss.masked_select(non_pad_mask).sum()
            nll_loss = nll_loss.masked_select(non_pad_mask).sum()

        return loss, nll_loss