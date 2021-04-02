import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, d_model, n_layers, src_pdx=0, tgt_pdx=0):
        super().__init__()
        self.encoder = Encoder(n_src_words, src_pdx, d_model, n_layers)
        self.decoder = Decoder(n_tgt_words, tgt_pdx, d_model, n_layers)
        self.out_vocab_proj = nn.Linear(d_model, n_tgt_words)
    
    def forward(self, src_tokens, prev_tgt_tokens):
        '''
        params:
          - src_tokens: (batch_size, src_len)
          - prev_tgt_tokens: (batch_size, tgt_len)
        returns:
          - model_out: (batch_size, tgt_len, n_tgt_words)
        '''

        hidden =  self.encoder(src_tokens)
        decoder_out = self.decoder(prev_tgt_tokens, hidden)
        model_out = self.out_vocab_proj(decoder_out)
        return model_out

class Encoder(nn.Module):
    def __init__(self, n_src_words, src_pdx, d_model, n_layers):
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers
        self.input_embedding = nn.Embedding(n_src_words, d_model, padding_idx=src_pdx)
        self.rnn = nn.RNN(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                          batch_first=True)
    
    def forward(self, src_tokens):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens)
        h_0 = src_embed.new_zeros(self.n_layers, src_tokens.size(0), self.d_model)
        _, hidden = self.rnn(src_embed, h_0)
        return hidden

class Decoder(nn.Module):
    def __init__(self, n_tgt_words, tgt_pdx, d_model, n_layers):
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers
        self.input_embedding = nn.Embedding(n_tgt_words, d_model, padding_idx=tgt_pdx)
        self.rnn = nn.RNN(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                          batch_first=True)
    
    def forward(self, prev_tgt_tokens, hidden):
        tgt_embed = self.input_embedding(prev_tgt_tokens)
        decoder_out, _ = self.rnn(tgt_embed, hidden)
        return decoder_out