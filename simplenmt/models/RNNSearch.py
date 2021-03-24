import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2seq(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, max_src_len, max_tgt_len, d_model, n_layers, src_pdx=0, tgt_pdx=0):
        super().__init__()
        encoder = Encoder(n_src_words, max_src_len, d_model, src_pdx, n_layers)
        decoder = Decoder(n_tgt_words, max_tgt_len, d_model, tgt_pdx, n_layers)
    
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
    def __init__(self, n_src_words, max_src_len, d_model, src_pdx, n_layers):
        super().__init__()
        self.input_embedding = nn.Embedding(n_src_words, d_model, padding_idx=src_pdx)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, batch_first=True, bidirectional=True)
    
    def forward(self, src_tokens):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens)



        encoder_out = src_tokens
        return encoder_out

class Decoder(nn.Module):
    def __init__(self, max_tgt_len):
        super().__init__()
    
    def forward(self, prev_tgt_tokens, encoder_out):
        decoder_out = prev_tgt_tokens
        return decoder_out