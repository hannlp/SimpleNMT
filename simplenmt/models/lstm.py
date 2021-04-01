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
        src_mask = src_tokens.eq(self.src_pdx)
        encoder_out, hiddens, cells, src_mask =  self.encoder(src_tokens)
        decoder_out = self.decoder(prev_tgt_tokens, encoder_out, src_mask)
        model_out = decoder_out
        return model_out

class Encoder(nn.Module):
    def __init__(self, n_src_words, max_src_len, d_model, src_pdx, n_layers, bidirectional):
        super().__init__()
        self.d_model, self.n_layers, self.src_pdx = d_model, n_layers, src_pdx
        self.n_directions = 2 if bidirectional else 1
        self.input_embedding = nn.Embedding(n_src_words, d_model, padding_idx=src_pdx)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                            batch_first=True, bidirectional=bidirectional)
    
    def forward(self, src_tokens):
        batch_size, src_lens = src_tokens.size(0), src_tokens.ne(self.src_pdx).long().sum(dim=-1)
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens)
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(
            src_embed, src_lens.to('cpu'), batch_first=True, enforce_sorted=False
        )
        state_size = self.n_layers * self.n_directions, batch_size, self.d_model
        h_0, c_0 = src_embed.new_zeros(*state_size), src_embed.new_zeros(*state_size)
        # - h_0 & c_0: (n_layers * n_directions, batch_size, d_model)
        packed_encoder_out, (hiddens, cells) = self.lstm(packed_src_embed, (h_0, c_0))
        # - encoder_out: (batch_size, src_len, n_directions * d_model)
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_encoder_out, batch_first=True
        )
        if self.n_directions == 2:
            hiddens = self._combine_bidir(hiddens, batch_size)
            cells = self._combine_bidir(cells, batch_size)
        # - hiddens & cells: (n_layers, batch_size, n_directions * d_model)
        return encoder_out, hiddens, cells

    def _combine_bidir(self, outs, batch_size):
        out = outs.view(self.n_layers, 2, batch_size, -1).transpose(1, 2).contiguous()
        return out.view(self.n_layers, batch_size, -1)

class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, encoder_out, src_mask):
        pass

class Decoder(nn.Module):
    def __init__(self, n_tgt_words, max_tgt_len, d_model, tgt_pdx, n_layers, bidirectional):
        super().__init__()
        self.d_model, self.n_layers, self.src_pdx = d_model, n_layers, tgt_pdx
        self.n_directions = 2 if bidirectional else 1
        self.input_embedding = nn.Embedding(n_tgt_words, d_model, padding_idx=tgt_pdx)
        self.lstm = nn.LSTM(input_size=self.n_directions * d_model, hidden_size=d_model, num_layers=n_layers, 
                            batch_first=True, bidirectional=False)
    
    def forward(self, prev_tgt_tokens, hidden, src_mask):
        tgt_embed = self.input_embedding(prev_tgt_tokens)
        decoder_out, hidden = self.gru(tgt_embed, hidden)
        return decoder_out