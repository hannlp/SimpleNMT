import torch
import torch.nn as nn
from models import str2model

def build_model(args, use_cuda):

    model_args = {
        "Transformer":
        {
            'n_src_words': args.n_src_words,
            'n_tgt_words': args.n_tgt_words, 
            'src_pdx': args.src_pdx, 
            'tgt_pdx': args.tgt_pdx, 
            'd_model': args.d_model, 
            'n_head': args.n_head, 
            'n_layers':args.n_layers, 
            'p_drop':args.p_drop,
            'share_embeddings':args.share_vocab, 
            'share_decoder_embeddings':args.share_vocab,
            'max_seq_len':args.max_seq_len
        }, 
        "RNN":{}, 
        "CNN":{}
        }
    
    MODEL = str2model[args.model]
    model = MODEL(**model_args[args.model])

    if use_cuda:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model