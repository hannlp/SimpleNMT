import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, d_model, n_layers, src_pdx=0, tgt_pdx=0, p_drop=0.1):
        super().__init__()
        self.encoder = Encoder(n_src_words, src_pdx, d_model, n_layers, p_drop)
        self.decoder = Decoder(n_tgt_words, tgt_pdx, d_model, n_layers, p_drop)
        self.out_vocab_proj = nn.Linear(d_model, n_tgt_words)

        # We initialized all of the LSTM’s parameters with the uniform distribution between -0.08 and 0.08
        for _, params in self.named_parameters():
            nn.init.uniform_(params.data, -0.08, 0.08)
    
    def forward(self, src_tokens, prev_tgt_tokens):
        '''
        params:
          - src_tokens: (batch_size, src_len)
          - prev_tgt_tokens: (batch_size, tgt_len)
        returns:
          - model_out: (batch_size, tgt_len, n_tgt_words)
        '''

        hiddens, cells =  self.encoder(src_tokens)
        decoder_out = self.decoder(prev_tgt_tokens, hiddens, cells)
        model_out = self.out_vocab_proj(decoder_out)
        return model_out

class Encoder(nn.Module):
    def __init__(self, n_src_words, src_pdx, d_model, n_layers, p_drop):
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers
        self.input_embedding = nn.Embedding(n_src_words, d_model, padding_idx=src_pdx)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                          dropout=p_drop, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p=p_drop)
    
    def forward(self, src_tokens):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.dropout(self.input_embedding(src_tokens))
        _, (hiddens, cells) = self.lstm(src_embed)
        # - hiddens & cells: (n_layers, batch_size, d_model)
        return hiddens, cells

class Decoder(nn.Module):
    def __init__(self, n_tgt_words, tgt_pdx, d_model, n_layers, p_drop):
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers
        self.input_embedding = nn.Embedding(n_tgt_words, d_model, padding_idx=tgt_pdx)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                          dropout=p_drop, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, prev_tgt_tokens, hiddens, cells):
        tgt_embed = self.dropout(self.input_embedding(prev_tgt_tokens))
        decoder_out, _ = self.lstm(tgt_embed, (hiddens, cells))
        return decoder_out