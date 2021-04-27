import os
import logging
import torch.nn as nn
from .loss import LabelSmoothingLoss

def build_criterion(args):
    if args.label_smoothing > 0:
        criterion = LabelSmoothingLoss(args.label_smoothing, ignore_index=args.tgt_pdx, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.tgt_pdx, reduction='mean')
    return criterion

def get_logger(args):
    log_path = args.save_path + '/log.txt'
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    consle_handler = logging.StreamHandler()
    consle_handler.setLevel(logging.INFO)
    consle_handler.setFormatter(formatter)

    logger.addHandler(consle_handler)
    logger.addHandler(file_handler)
    return logger