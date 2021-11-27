import torch
import torch.nn as nn
 # debugging: use bias=False in out_vocab_proj
#from .transformer_dev import Transformer
from .transformer import Transformer
from .rnnsearch import RNNSearch
from .seq2seq import Seq2Seq
from .luong import Luong
from .convs2s import ConvS2S
from .transformer_v2 import Transformer2


str2model = {"Transformer": Transformer, "RNNSearch": RNNSearch, "Seq2Seq": Seq2Seq, 
                "Luong": Luong, "ConvS2S": ConvS2S, "Transformer2": Transformer2}


def build_model(args, training=True):

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
            'max_seq_len': args.max_seq_len
            },
        "Transformer2": {
            'n_src_words': args.n_src_words,
            'n_tgt_words': args.n_tgt_words,
            'src_pdx': args.src_pdx,
            'tgt_pdx': args.tgt_pdx,
            'd_model': args.d_model,
            'd_ff': args.d_ff,
            'n_head': args.n_head,
            'n_encoder_layers': args.n_encoder_layers,
            'n_decoder_layers': args.n_decoder_layers,
            'p_drop': args.p_drop,
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

    if args.share_vocab: 
        model.out_vocab_proj.weight = model.decoder.input_embedding.weight
        model.encoder.input_embedding.weight = model.decoder.input_embedding.weight

    if args.use_cuda:
        model.cuda()
        if torch.cuda.device_count() > 1 and training:
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