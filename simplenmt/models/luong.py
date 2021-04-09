import torch
import torch.nn as nn
import torch.nn.functional as F

class Luong(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, d_model, n_layers, 
                 src_pdx=-1, tgt_pdx=-1, p_drop=0.2, bidirectional=False):
        super().__init__()
        self.src_pdx = src_pdx
        self.encoder = Encoder(n_src_words, d_model, src_pdx, n_layers, p_drop, bidirectional)
        self.decoder = Decoder(n_tgt_words, d_model, tgt_pdx, n_layers, p_drop, bidirectional)
        self.out_vocab_proj = nn.Linear(d_model, n_tgt_words)

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
        model_out = self.out_vocab_proj(decoder_out)
        return model_out

class Encoder(nn.Module):
    def __init__(self, n_src_words, d_model, src_pdx, n_layers, p_drop, bidirectional):
        super().__init__()
        self.d_model, self.n_layers, self.src_pdx = d_model, n_layers, src_pdx
        self.n_directions = 2 if bidirectional else 1
        self.input_embedding = nn.Embedding(n_src_words, d_model, padding_idx=src_pdx)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                            dropout=p_drop, batch_first=True, bidirectional=bidirectional)
    
    def forward(self, src_tokens):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens)
        batch_size, src_lens = src_tokens.size(0), src_tokens.ne(self.src_pdx).long().sum(dim=-1)
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(
            src_embed, src_lens.to('cpu'), batch_first=True, enforce_sorted=False
        )
        
        packed_encoder_out, (hiddens, cells) = self.lstm(packed_src_embed)

        # - encoder_out: (batch_size, src_len, n_directions * d_model) where 3rd is last layer [h_fwd; (h_bkwd)]
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(packed_encoder_out, batch_first=True)

        # - hiddens & cells: (n_layers, batch_size, n_directions * d_model)
        if self.n_directions == 2:
            hiddens = self._combine_bidir(hiddens, batch_size)
            cells = self._combine_bidir(cells, batch_size)

        return encoder_out, hiddens, cells

    def _combine_bidir(self, outs, batch_size):
        out = outs.view(self.n_layers, 2, batch_size, -1).transpose(1, 2).contiguous()
        return out.view(self.n_layers, batch_size, -1)

class AttentionLayer(nn.Module):
    # general attention in 2015 luong et.
    def __init__(self, d_src, d_tgt):
        super().__init__()
        self.linear_in = nn.Linear(d_tgt, d_src, bias=False)

    def forward(self, source, memory, mask):
        # - source: (batch_size, tgt_len, d_tgt), memory: (batch_size, src_len, d_src)
        score = torch.matmul(self.linear_in(source), memory.transpose(1, 2))
        score.masked_fill_(mask, -1e9)
        # - score: (batch_size, tgt_len, src_len)
        attn = F.softmax(score, dim=-1)
        return attn

class Decoder(nn.Module):
    def __init__(self, n_tgt_words, d_model, tgt_pdx, n_layers, p_drop, bidirectional):
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers
        self.n_directions = 2 if bidirectional else 1
        self.input_embedding = nn.Embedding(n_tgt_words, d_model, padding_idx=tgt_pdx)

        self.attention = AttentionLayer(d_src=self.n_directions * d_model, d_tgt=d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                            dropout=p_drop, batch_first=True, bidirectional=False)

        self.linear_out = nn.Linear(self.n_directions * d_model + d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, prev_tgt_tokens, encoder_out, hiddens, cells, src_mask):
        # - tgt_embed: (batch_size, src_len, d_model)
        tgt_embed = self.input_embedding(prev_tgt_tokens)

        # - decoder_states: (batch_size, tgt_len, d_model)
        decoder_states, _ = self.lstm(tgt_embed)

        attn = self.attention(source=decoder_states, memory=encoder_out, mask=src_mask.unsqueeze(1))
        # - attn: (batch_size, tgt_len, src_len), encoder_out: (batch_size, src_len, d_src)
 
        context = torch.matmul(attn, encoder_out)
        # - context: (batch_size, tgt_len, d_src)
        
        decoder_out = self.dropout(self.linear_out(torch.cat([context, decoder_states], dim=-1)))
        # - decoder_out: (batch_size, tgt_len, d_model)
        return decoder_out