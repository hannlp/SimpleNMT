import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, max_src_len, max_tgt_len,
                 d_model, n_layers, src_pdx=0, tgt_pdx=0, bidirectional=True):
        super().__init__()
        self.encoder = Encoder(n_src_words, max_src_len, d_model, src_pdx, n_layers, bidirectional)
        self.decoder = Decoder(n_tgt_words, max_tgt_len, d_model, tgt_pdx, n_layers)
    
    def forward(self, src_tokens, prev_tgt_tokens):
        '''
        params:
          - src_tokens: (batch_size, src_len)
          - prev_tgt_tokens: (batch_size, tgt_len)
        returns:
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
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                            batch_first=True, bidirectional=bidirectional)
    
    def forward(self, src_tokens):
        batch_size, src_len = src_tokens.size()
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens)
        state_size = self.n_layers * self.n_directions, batch_size,  self.d_model
        h_0 = src_embed.new_zeros(*state_size)
        c_0 = src_embed.new_zeros(*state_size)
        encoder_out, (hiddens, cells) = self.lstm(src_embed, (h_0, c_0))
        if self.n_directions == 2:
            hiddens = self._combine_bidir(hiddens, batch_size)
            cells = self._combine_bidir(cells, batch_size)
        return encoder_out, hiddens, cells

    def _combine_bidir(self, outs, batch_size: int):
        out = outs.view(self.n_layers, 2, batch_size, -1).transpose(1, 2).contiguous()
        return out.view(self.n_layers, batch_size, -1)

class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Decoder(nn.Module):
    def __init__(self, n_tgt_words, max_tgt_len, d_model, tgt_pdx, n_layers):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.input_embedding = nn.Embedding(n_tgt_words, d_model, padding_idx=tgt_pdx)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                          batch_first=True)
    
    def forward(self, prev_tgt_tokens, hidden):
        tgt_embed = self.input_embedding(prev_tgt_tokens)
        decoder_out, hidden = self.gru(tgt_embed, hidden)
        return decoder_out