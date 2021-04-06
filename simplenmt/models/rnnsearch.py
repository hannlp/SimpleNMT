import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNSearch(nn.Module):
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
        encoder_out, hiddens, cells =  self.encoder(src_tokens)
        decoder_out = self.decoder(prev_tgt_tokens, encoder_out, hiddens, cells, src_mask)
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
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens)
        batch_size, src_lens = src_tokens.size(0), src_tokens.ne(self.src_pdx).long().sum(dim=-1)
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(
            src_embed, src_lens.to('cpu'), batch_first=True, enforce_sorted=False
        )

        # - h_0 & c_0: (n_layers * n_directions, batch_size, d_model)
        state_size = self.n_layers * self.n_directions, batch_size, self.d_model
        h_0, c_0 = [src_embed.new_zeros(*state_size)] * 2
        
        packed_encoder_out, (hiddens, cells) = self.lstm(packed_src_embed, (h_0, c_0))
        # - encoder_out: (batch_size, src_len, n_directions * d_model) where 3rd is last layer [h_fwd; h_bkwd]
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_encoder_out, batch_first=True
        )

        # - hiddens & cells: (n_layers, batch_size, n_directions * d_model)
        if self.n_directions == 2:
            hiddens = self._combine_bidir(hiddens, batch_size)
            cells = self._combine_bidir(cells, batch_size)
        
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

        self.layers = nn.ModuleList(
            [nn.LSTMCell(input_size=d_model, hidden_size=d_model) for _ in range(n_layers)])
        self.attention = AttentionLayer()
        # self.lstm = nn.LSTM(input_size=self.n_directions * d_model, hidden_size=d_model, num_layers=n_layers, 
        #                     batch_first=True, bidirectional=False)
    
    def forward(self, prev_tgt_tokens, encoder_out, hiddens, cells, src_mask):
        tgt_len = prev_tgt_tokens.size(1)
        # - tgt_embed: (batch_size, src_len, d_model)
        tgt_embed = self.input_embedding(prev_tgt_tokens)
        prev_hiddens, prev_cells = [hiddens[l] for l in range(self.n_layers)], [cells[l] for l in range(self.n_layers)]
        #batch_size, src_len, tgt_len = encoder_out.size[:-1], prev_tgt_tokens.size(1)
        #attn_scores = tgt_embed.new_zeros(batch_size, src_len, tgt_len)
        for t in range(tgt_len):
            # - y_t: (batch_size, d_model)
            y_t = tgt_embed[:, t, :]
            s = y_t
            for l, layer in enumerate(self.layers):
                hidden, cell = layer(s, (prev_hiddens[l], prev_cells[l]))
                s = hidden
                prev_hiddens[l], prev_hiddens[l] = hidden, cell
            # - s_t: (batch_size, d_model)

            out = self.attention(s, encoder_out, src_mask)
            #s_t = f(s_t-1, y_t-1, c_t)

            


        return decoder_out