import torch.nn as nn
from .loss import LabelSmoothingLoss

def build_criterion(args):
    if args.label_smoothing > 0:
        criterion = LabelSmoothingLoss(args.label_smoothing, ignore_index=args.tgt_pdx, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.tgt_pdx, reduction='mean')
    return criterion