import torch
import torch.nn as nn
import torch.nn.functional as F

str2rnn = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

class Luong(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, d_model, n_layers, src_pdx=-1, tgt_pdx=-1, 
                 p_drop=0.2, bidirectional=False, attn_type='general', rnn_type='lstm'):
        super().__init__()
        self.src_pdx = src_pdx
        self.encoder = Encoder(n_src_words, d_model, src_pdx, n_layers, p_drop, 
                               bidirectional, rnn_type=rnn_type)
        self.decoder = Decoder(n_tgt_words, d_model, tgt_pdx, n_layers, p_drop, 
                               attn_type, rnn_type=rnn_type)
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
        encoder_out =  self.encoder(src_tokens)
        decoder_out = self.decoder(prev_tgt_tokens, encoder_out, src_mask)
        model_out = self.out_vocab_proj(decoder_out)
        return model_out


class Encoder(nn.Module):
    def __init__(self, n_src_words, d_model, src_pdx, n_layers, p_drop, bidirectional, rnn_type):
        super().__init__()
        self.d_model, self.n_layers, self.src_pdx = d_model, n_layers, src_pdx
        self.n_directions = 2 if bidirectional else 1
        self.input_embedding = nn.Embedding(n_src_words, d_model, padding_idx=src_pdx)
        
        self.rnn = str2rnn[rnn_type](input_size=d_model, hidden_size=d_model // self.n_directions, 
                            num_layers=n_layers, dropout=p_drop, 
                            batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=p_drop)
    
    def forward(self, src_tokens):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.dropout(self.input_embedding(src_tokens))
        src_lens = src_tokens.ne(self.src_pdx).long().sum(dim=-1)
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(
            src_embed, src_lens.to('cpu'), batch_first=True, enforce_sorted=False
        )
        
        packed_encoder_out, _ = self.rnn(packed_src_embed)

        # - encoder_out: (batch_size, src_len, d_model) where 3rd is last layer [h_fwd; (h_bkwd)]
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(packed_encoder_out, batch_first=True)
        return encoder_out

class AttentionLayer(nn.Module):
    # 2015 luong et, Effective Approaches to Attention-based Neural Machine Translation
    def __init__(self, d_model, attn_type='general'):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == 'dot':
            pass
        elif attn_type == 'general':
            self.W_align = nn.Linear(d_model, d_model, bias=False)
        elif attn_type == 'concat':
            self.W_align_source = nn.Linear(d_model, d_model, bias=False)
            self.W_align_memory = nn.Linear(d_model, d_model, bias=False)
            self.v_align = nn.Linear(d_model, 1, bias=False)
        else:
            raise Exception
    
    def forward(self, source, memory, mask=None):
        # - source: (batch_size, tgt_len, d_model), memory: (batch_size, src_len, d_model)
        batch_size, src_len, tgt_len = memory.size(0), memory.size(1), source.size(1)

        if self.attn_type == 'dot':
            score = torch.matmul(source, memory.transpose(1, 2))
        elif self.attn_type == 'general':
            score = torch.matmul(self.W_align(source), memory.transpose(1, 2))
        elif self.attn_type == 'concat':
            # (batch_size, tgt_len, d_model) can't directly concat with (batch_size, src_len, d_model)
            source = self.W_align_source(
                source.view(batch_size, tgt_len, 1, -1).expand(batch_size, tgt_len, src_len, -1))
            memory = self.W_align_memory(
                memory.view(batch_size, 1, src_len, -1).expand(batch_size, tgt_len, src_len, -1))
            score = self.v_align(source + memory).view(batch_size, tgt_len, src_len)
        else:
            raise Exception
        
        # - score: (batch_size, tgt_len, src_len)
        if mask != None:
            score.masked_fill_(mask, -1e9)
        
        attn = F.softmax(score, dim=-1)
        return attn

class Decoder(nn.Module):
    def __init__(self, n_tgt_words, d_model, tgt_pdx, n_layers, p_drop, attn_type, rnn_type):
        super().__init__()
        self.d_models = d_model
        self.input_embedding = nn.Embedding(n_tgt_words, d_model, padding_idx=tgt_pdx)
        self.attention = AttentionLayer(d_model=d_model, attn_type=attn_type)
        self.rnn = str2rnn[rnn_type](input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                            dropout=p_drop, batch_first=True, bidirectional=False)

        self.W_context = nn.Linear(2 * d_model, d_model, bias=False) # for concat [c; h]
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, prev_tgt_tokens, encoder_out, src_mask):
        # - tgt_embed: (batch_size, src_len, d_model)
        tgt_embed = self.dropout(self.input_embedding(prev_tgt_tokens))

        # - decoder_states: (batch_size, tgt_len, d_model)
        decoder_states, _ = self.rnn(tgt_embed)

        # - attn: (batch_size, tgt_len, src_len), encoder_out: (batch_size, src_len, d_model)
        attn = self.attention(source=decoder_states, memory=encoder_out, mask=src_mask.unsqueeze(1))
        
        # - context: (batch_size, tgt_len, d_model)
        context = torch.matmul(attn, encoder_out)
        
        # - decoder_out: (batch_size, tgt_len, d_model)
        decoder_out = self.dropout(self.W_context(torch.cat([context, decoder_states], dim=-1)))
        return decoder_out