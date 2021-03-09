import torch
import torch.nn as nn
from models.transformer import Transformer


'''
MODEL PARAMETERS:
Transformer(
    n_src_words, 
    n_tgt_words, 
    src_pdx=0, 
    tgt_pdx=0, 
    d_model=512, 
    n_head=8, 
    n_layer=6, 
    p_drop=0.1
)
'''


def build_model(args, MODEL, CUDA_OK=False):
    
    # 
    args_dict = {}

    model = MODEL(**args_dict)
    if CUDA_OK:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model

#print(dir())