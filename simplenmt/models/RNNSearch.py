import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNSearch(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, max_src_len, max_tgt_len, d_model, n_layers, src_pdx=0, tgt_pdx=0, bidirectional=True):
        super().__init__()
        self.encoder = Encoder(n_src_words, max_src_len, d_model, src_pdx, n_layers, bidirectional)
        self.decoder = Decoder(n_tgt_words, max_tgt_len, d_model, tgt_pdx, n_layers)
    
    def forward(self, src_tokens, prev_tgt_tokens):
        '''
        params:
          - src_tokens: (batch_size, src_len)
          - prev_tgt_tokens: (batch_size, tgt_len)
        return:
          - model_out: (batch_size, tgt_len, n_tgt_words)
        '''

        _, hidden =  self.encoder(src_tokens)
        decoder_out = self.decoder(prev_tgt_tokens, hidden)
        model_out = decoder_out       
        return model_out

class Encoder(nn.Module):
    def __init__(self, n_src_words, max_src_len, d_model, src_pdx, n_layers, bidirectional):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.input_embedding = nn.Embedding(n_src_words, d_model, padding_idx=src_pdx)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                          batch_first=True, bidirectional=bidirectional)
    
    def forward(self, src_tokens):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens)
        init_hidden = torch.zeros((self.n_layers * self.n_directions), src_tokens.size(0),  self.d_model)
        encoder_out, hidden = self.gru(src_embed, init_hidden)
        return encoder_out, hidden

class Decoder(nn.Module):
    def __init__(self, n_tgt_words, max_tgt_len, d_model, tgt_pdx, n_layers):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.input_embedding = nn.Embedding(n_tgt_words, d_model, padding_idx=tgt_pdx)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                          batch_first=True)
    
    def forward(self, prev_tgt_tokens, hidden):
        tgt_embed = self.input_embedding(prev_tgt_tokens)
        decoder_out, hidden = self.gru(tgt_embed, hidden)
        return decoder_out