import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2seq(nn.Module):
    def __init__(self, max_src_len, max_tgt_len):
        super().__init__()
        encoder = Encoder(max_src_len)
        decoder = Decoder(max_tgt_len)
    
    def forward(self, src_tokens, prev_tgt_tokens):
        '''
        params:
          - src_tokens: (batch_size, src_len)
          - prev_tgt_tokens: (batch_size, tgt_len)
        return:
          - model_out: (batch_size, tgt_len, n_tgt_words)
        '''

        encoder_out =  self.encoder(src_tokens)
        decoder_out = self.decoder(prev_tgt_tokens, encoder_out)
        model_out = decoder_out       
        return model_out

class Encoder(nn.Module):
    def __init__(self, max_src_len):
        super().__init__()
    
    def forward(self, src_tokens):
        encoder_out = src_tokens
        return encoder_out

class Decoder(nn.Module):
    def __init__(self, max_tgt_len):
        super().__init__()
    
    def forward(self, prev_tgt_tokens, encoder_out):
        decoder_out = prev_tgt_tokens
        return decoder_out