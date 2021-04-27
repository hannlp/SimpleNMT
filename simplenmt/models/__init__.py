import torch
import torch.nn as nn
from .transformer import Transformer
from .rnnsearch import RNNSearch
from .seq2seq import Seq2Seq
from .luong import Luong
from .convs2s import ConvS2S

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

str2model = {"Transformer": Transformer, "RNNSearch": RNNSearch, "Seq2Seq": Seq2Seq, "Luong": Luong, "ConvS2S": ConvS2S}

def build_model(args):

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
        "Luong": {
            'n_src_words': args.n_src_words,
            'n_tgt_words': args.n_tgt_words,
            'd_model': args.d_model, 
            'n_layers': args.n_layers, 
            'src_pdx': args.src_pdx, 
            'tgt_pdx': args.tgt_pdx,
            'p_drop': args.p_drop,
            'bidirectional': args.bidirectional,
            'attn_type': args.attn_type, 
            'rnn_type': args.rnn_type
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
        "ConvS2S": {

            }
        }
    
    MODEL = str2model[args.model]
    model = MODEL(**model_args[args.model])

    if args.use_cuda:
        model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
    return model

def count_parameters(model, logger):
    # Count number of parameters in model

    enc, dec, others = 0, 0, 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
        else:
            others += param.nelement()

    logger.info('Params count | encoder: {}, decoder: {}, others: {}, total: {}'.format(
        format(enc, ','), format(dec, ','), format(others, ','), format(enc + dec + others, ',')))