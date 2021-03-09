import torch
import torch.nn as nn
from models.transformer import Transformer
from models import str2model

def build_model(args, cuda_ok):

    model_args = {
        "Transformer":
        {
            'n_src_words': args.n_src_words,
            'n_tgt_words': args.n_tgt_words, 
            'src_pdx': args.src_pdx, 
            'tgt_pdx': args.tgt_pdx, 
            'd_model': args.d_model, 
            'n_head': args.n_head, 
            'n_layer':args.n_layer, 
            'p_drop':args.p_drop
        }, 
        "RNN":{}, 
        "CNN":{}
        }
    
    MODEL = str2model[args.model]
    model = MODEL(**model_args[args.model])

    if cuda_ok:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    print(model)
    return model