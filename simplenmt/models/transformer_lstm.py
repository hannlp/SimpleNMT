import torch
import torch.nn as nn
import torch.nn.functional as F

"""
 A combine version of transformer and lstm,
 用Transformer编码，用LSTM解码，不知道能不能work，试一下
"""

class Transformer_LSTM(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, src_pdx=-1, tgt_pdx=-1, d_model=512, d_ff=2048, n_head=8, 
                 n_layers=6, n_encoder_layers=None, n_decoder_layers=None, p_drop=0.1, max_seq_len=512) -> None:

        super().__init__()
        self.d_model = d_model
        self.src_pdx, self.tgt_pdx = src_pdx, tgt_pdx  # pdx: padding index
        
        self.encoder = Encoder(n_src_words, src_pdx=src_pdx, n_head=n_head, d_model=d_model, d_ff=d_ff, 
                               n_layers=n_layers if n_encoder_layers is None else n_encoder_layers, 
                               p_drop=p_drop, max_seq_len=max_seq_len)

        self.decoder = Decoder(n_tgt_words, tgt_pdx=tgt_pdx, n_head=n_head, d_model=d_model, p_drop=p_drop,
                               n_layers=n_layers if n_decoder_layers is None else n_decoder_layers)
        self.out_vocab_proj = nn.Linear(d_model, n_tgt_words)
        
        self._model_init()

    def forward(self, src_tokens, prev_tgt_tokens):
        '''
        params:
          - src_tokens: (batch_size, src_len)
          - prev_tgt_tokens: (batch_size, tgt_len)
        
        returns:
          - model_out: (batch_size, tgt_len, n_tgt_words)
        '''

        src_mask = src_tokens.eq(self.src_pdx)

        encoder_out = self.encoder(src_tokens, src_mask)
        decoder_out = self.decoder(
            prev_tgt_tokens, encoder_out, src_mask)
        model_out = self.out_vocab_proj(decoder_out)
        return model_out

    def _model_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Encoder(nn.Module):
    def __init__(self, n_src_words, src_pdx, n_head, d_model, d_ff,
                 n_layers, p_drop, max_seq_len) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=p_drop)
        self.input_embedding = nn.Embedding(
            num_embeddings=n_src_words, embedding_dim=d_model, padding_idx=src_pdx)
        self.positional_encode = PositionalEncode(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, n_head, p_drop) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model) # for memory

    def forward(self, src_tokens, src_mask, **kwargs):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens) * (self.d_model ** 0.5)
        x = self.dropout(self.positional_encode(src_embed))
        for layer in self.layers:
            x = layer(x, src_mask)
        encoder_out = self.layer_norm(x)
        return encoder_out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, p_drop) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)
        self.prenorm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.prenorm2 = nn.LayerNorm(d_model)
        self.pos_wise_ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, x, src_mask):
        x_norm = self.prenorm1(x)
        x = x + self.dropout(self.self_attn(
            q=x_norm, k=x_norm, v=x_norm, mask=src_mask.unsqueeze(1).unsqueeze(1)))
        x_norm = self.prenorm2(x)
        x = x + self.dropout(self.pos_wise_ffn(x_norm))
        return x

class Decoder(nn.Module):
    def __init__(self, n_tgt_words, n_head, d_model, tgt_pdx, n_layers, p_drop):
        super().__init__()
        self.d_models = d_model
        self.input_embedding = nn.Embedding(n_tgt_words, d_model, padding_idx=tgt_pdx)
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.rnn = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                            dropout=p_drop, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, prev_tgt_tokens, encoder_out, src_mask, **kwargs):
        # - tgt_embed: (batch_size, tgt_len, d_model)
        tgt_embed = self.dropout(self.input_embedding(prev_tgt_tokens))

        # - decoder_states: (batch_size, tgt_len, d_model)
        decoder_states, _ = self.rnn(tgt_embed)

        # - O: (batch_size, tgt_len, d_model), encoder_out: (batch_size, src_len, d_model)
        O = self.attention(q=decoder_states, k=encoder_out, v=encoder_out, mask=src_mask.unsqueeze(1).unsqueeze(1))

        # - decoder_out: (batch_size, tgt_len, d_model)
        decoder_out = self.dropout(O)
        return decoder_out

class MultiHeadAttention(nn.Module):
    # - src_embed_dim = d_model
    def __init__(self, d_model, n_head) -> None:
        super().__init__()
        self.n_head, self.one_head_dim = n_head, d_model // n_head
        self.w_q = nn.Linear(d_model, self.one_head_dim * self.n_head, bias=True)
        self.w_k = nn.Linear(d_model, self.one_head_dim * self.n_head, bias=True)
        self.w_v = nn.Linear(d_model, self.one_head_dim * self.n_head, bias=True)
        self.w_out = nn.Linear(self.one_head_dim * self.n_head, d_model, bias=True)

    def forward(self, q, k, v, mask=None):
        # - x: (batch_size, seq_len, d_model)
        batch_size, q_len, kv_len = q.size(0), q.size(1), k.size(1)
        Q = self.w_q(q).view(batch_size, q_len, self.n_head, 
                             self.one_head_dim).transpose(1, 2)
        K = self.w_k(k).view(batch_size, kv_len, self.n_head,
                             self.one_head_dim).transpose(1, 2)
        V = self.w_v(v).view(batch_size, kv_len, self.n_head,
                             self.one_head_dim).transpose(1, 2)
        # - Q, K, V: (batch_size, n_head, seq_len, one_head_dim)

        Q_KT = torch.matmul(Q, torch.transpose(K, 2, 3))

        if mask != None:
            Q_KT.masked_fill_(mask, -1e9)

        attn = F.softmax(Q_KT / self.one_head_dim ** 0.5, dim=-1)

        O = self.w_out(torch.matmul(attn, V).transpose(1, 2).reshape(
                batch_size, q_len, self.one_head_dim * self.n_head))
        # - O: (batch_size, seq_len, d_model)
        return O


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class PositionalEncode(nn.Module):
    def __init__(self, d_model, max_seq_len=512) -> None:
        super().__init__()
        self.pos_encode = self._get_pos_encode(max_seq_len, d_model)

    def forward(self, x):
        # - x: (batch_size, seq_len, d_model)
        return x + self.pos_encode[:x.size(1), :].unsqueeze(0).to(x.device)

    def _get_pos_encode(self, max_seq_len, d_model):
        # TODO: 尝试使用矩阵乘法，观察哪种方式速度更快
        pos_encode = torch.tensor([[pos / 10000 ** (2 * (i//2) / d_model) for i in range(d_model)]
                                   for pos in range(max_seq_len)], requires_grad=False)
        pos_encode[:, 0::2] = torch.sin(pos_encode[:, 0::2])
        pos_encode[:, 1::2] = torch.cos(pos_encode[:, 1::2])
        # - pos_encode: (seq_len, d_model)
        return pos_encode

"""
class Transformer_LSTM(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, src_pdx=-1, tgt_pdx=-1, 
                 d_model=512, n_head=8, n_layers=6, p_drop=0.1, attn_type='general', max_seq_len=512) -> None:

        super().__init__()
        self.d_model = d_model
        self.src_pdx, self.tgt_pdx = src_pdx, tgt_pdx  # pdx: padding index
        
        self.encoder = Encoder(n_src_words, src_pdx=src_pdx, n_head=n_head, 
                               d_model=d_model, n_layers=n_layers, p_drop=p_drop, 
                               max_seq_len=max_seq_len)

        self.decoder = Decoder(n_tgt_words, d_model, tgt_pdx, n_layers, p_drop, attn_type)
        self.out_vocab_proj = nn.Linear(d_model, n_tgt_words)
        
        self._model_init()

    def forward(self, src_tokens, prev_tgt_tokens):
        '''
        params:
          - src_tokens: (batch_size, src_len)
          - prev_tgt_tokens: (batch_size, tgt_len)
        
        returns:
          - model_out: (batch_size, tgt_len, n_tgt_words)
        '''

        src_mask = src_tokens.eq(self.src_pdx)
        encoder_out = self.encoder(src_tokens, src_mask)
        decoder_out = self.decoder(
            prev_tgt_tokens, encoder_out, src_mask)
        model_out = self.out_vocab_proj(decoder_out)
        return model_out

    def _model_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Encoder(nn.Module):
    def __init__(self, n_src_words, src_pdx, n_head, d_model, 
                 n_layers, p_drop, max_seq_len) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=p_drop)
        self.input_embedding = nn.Embedding(
            num_embeddings=n_src_words, embedding_dim=d_model, padding_idx=src_pdx)
        self.positional_encode = PositionalEncode(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, p_drop) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model) # for memory

    def forward(self, src_tokens, src_mask, **kwargs):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens) * (self.d_model ** 0.5)
        x = self.dropout(self.positional_encode(src_embed))
        for layer in self.layers:
            x = layer(x, src_mask)
        encoder_out = self.layer_norm(x)
        return encoder_out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, p_drop) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)
        self.sublayer1_prenorm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.sublayer2_prenorm = nn.LayerNorm(d_model)
        self.pos_wise_ffn = FeedForwardNetwork(d_model)

    def forward(self, x, src_mask):
        res, x_ln = x, self.sublayer1_prenorm(x)
        x = res + self.dropout(self.self_attn(
            q=x_ln, k=x_ln, v=x_ln,
            mask=src_mask.unsqueeze(1).unsqueeze(1)))
        res, x_ln = x, self.sublayer2_prenorm(x)
        x = res + self.dropout(self.pos_wise_ffn(x_ln))
        return x


class MultiHeadAttention(nn.Module):
    # - src_embed_dim = d_model
    def __init__(self, d_model, n_head) -> None:
        super().__init__()
        self.n_head, self.one_head_dim = n_head, d_model // n_head
        self.w_q = nn.Linear(d_model, self.one_head_dim * self.n_head, bias=True)
        self.w_k = nn.Linear(d_model, self.one_head_dim * self.n_head, bias=True)
        self.w_v = nn.Linear(d_model, self.one_head_dim * self.n_head, bias=True)
        self.w_out = nn.Linear(self.one_head_dim * self.n_head, d_model, bias=True)

    def forward(self, q, k, v, mask=None):
        # - x: (batch_size, seq_len, d_model)
        batch_size, q_len, kv_len = q.size(0), q.size(1), k.size(1)
        Q = self.w_q(q).view(batch_size, q_len, self.n_head, 
                             self.one_head_dim).transpose(1, 2)
        K = self.w_k(k).view(batch_size, kv_len, self.n_head,
                             self.one_head_dim).transpose(1, 2)
        V = self.w_v(v).view(batch_size, kv_len, self.n_head,
                             self.one_head_dim).transpose(1, 2)
        # - Q, K, V: (batch_size, n_head, seq_len, one_head_dim)

        Q_KT = torch.matmul(Q, torch.transpose(K, 2, 3))

        if mask != None:
            Q_KT.masked_fill_(mask, -1e9)

        attn = F.softmax(Q_KT / self.one_head_dim ** 0.5, dim=-1)

        O = self.w_out(torch.matmul(attn, V).transpose(1, 2).reshape(
                batch_size, q_len, self.one_head_dim * self.n_head))
        # - O: (batch_size, seq_len, d_model)
        return O


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model, bias=True)
        self.linear2 = nn.Linear(4 * d_model, d_model, bias=True)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class PositionalEncode(nn.Module):
    def __init__(self, d_model, max_seq_len=512) -> None:
        super().__init__()
        self.pos_encode = self._get_pos_encode(max_seq_len, d_model)

    def forward(self, x):
        # - x: (batch_size, seq_len, d_model)
        return x + self.pos_encode[:x.size(1), :].unsqueeze(0).to(x.device)

    def _get_pos_encode(self, max_seq_len, d_model):
        # TODO: 尝试使用矩阵乘法，观察哪种方式速度更快
        pos_encode = torch.tensor([[pos / 10000 ** (2 * (i//2) / d_model) for i in range(d_model)]
                                   for pos in range(max_seq_len)], requires_grad=False)
        pos_encode[:, 0::2] = torch.sin(pos_encode[:, 0::2])
        pos_encode[:, 1::2] = torch.cos(pos_encode[:, 1::2])
        # - pos_encode: (seq_len, d_model)
        return pos_encode

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
    def __init__(self, n_tgt_words, d_model, tgt_pdx, n_layers, p_drop, attn_type):
        super().__init__()
        self.d_models = d_model
        self.input_embedding = nn.Embedding(n_tgt_words, d_model, padding_idx=tgt_pdx)
        self.attention = AttentionLayer(d_model=d_model, attn_type=attn_type)
        self.rnn = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                            dropout=p_drop, batch_first=True, bidirectional=False)
        self.W_context = nn.Linear(2 * d_model, d_model, bias=False) # for concat [c; h]
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, prev_tgt_tokens, encoder_out, src_mask, **kwargs):
        # - tgt_embed: (batch_size, tgt_len, d_model)
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
"""