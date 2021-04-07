import torch
import torch.nn as nn
from .transformer import Transformer
from .rnnsearch import RNNSearch
from .seq2seq import Seq2Seq

'''
transformer:

1. use decoder last layer norm (deleted) 
2. move decoder vocab proj to transformer
3. add share vocab function
4. add max_seq_len args

3/16 update
5. change the mask compute method: != to .eq(), at : transforemr forward, _add_sequence_mask(), and maksed_fill()

3/24
6. change 'sequence' to 'subsequent'

'''

str2model = {"Transformer": Transformer, "RNNSearch": RNNSearch, "Seq2Seq": Seq2Seq}

def build_model(args, use_cuda):

    model_args = {
        "Transformer": {
            'n_src_words': args.n_src_words,
            'n_tgt_words': args.n_tgt_words, 
            'src_pdx': args.src_pdx, 
            'tgt_pdx': args.tgt_pdx, 
            'd_model': args.d_model, 
            'n_head': args.n_head, 
            'n_layers': args.n_layers, 
            'p_drop': args.p_drop,
            'share_embeddings': args.share_vocab, 
            'share_decoder_embeddings': args.share_vocab,
            'max_seq_len': args.max_seq_len
            }, 
        "RNNSearch": {
            'n_src_words': args.n_src_words,
            'n_tgt_words': args.n_tgt_words,
            'd_model': args.d_model, 
            'n_layers': args.n_layers, 
            'src_pdx': args.src_pdx, 
            'tgt_pdx': args.tgt_pdx,
            'max_src_len': args.max_seq_len,
            'max_tgt_len': args.max_seq_len
            }, 
        "Seq2Seq": {
            'n_src_words': args.n_src_words,
            'n_tgt_words': args.n_tgt_words,
            'd_model': args.d_model, 
            'n_layers': args.n_layers, 
            'src_pdx': args.src_pdx, 
            'tgt_pdx': args.tgt_pdx
            },
        "CNN": {

            }
        }
    
    MODEL = str2model[args.model]
    model = MODEL(**model_args[args.model])

    if use_cuda:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model