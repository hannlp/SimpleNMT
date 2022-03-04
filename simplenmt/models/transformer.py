import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, src_pdx=-1, tgt_pdx=-1,
                 d_model=512, d_ff=2048, n_head=8, n_encoder_layers=6,
                 n_decoder_layers=6, p_drop=0.1, max_seq_len=512,
                 encoder_prenorm=False, decoder_prenorm=False):

        super().__init__()
        self.d_model = d_model
        self.src_pdx, self.tgt_pdx = src_pdx, tgt_pdx

        self.encoder = Encoder(n_src_words, src_pdx=src_pdx, n_head=n_head,
                               d_model=d_model, d_ff=d_ff, n_layers=n_encoder_layers,
                               p_drop=p_drop, max_seq_len=max_seq_len,
                               encoder_prenorm=encoder_prenorm)

        self.decoder = Decoder(n_tgt_words, tgt_pdx=tgt_pdx, n_head=n_head,
                               d_model=d_model, d_ff=d_ff, n_layers=n_decoder_layers,
                               p_drop=p_drop, max_seq_len=max_seq_len,
                               decoder_prenorm=decoder_prenorm)

        self.out_vocab_proj = nn.Linear(d_model, n_tgt_words)

        self._model_init(encoder_prenorm, decoder_prenorm,
                         N=n_encoder_layers,
                         M=n_decoder_layers)

    def forward(self, src_tokens, prev_tgt_tokens):
        '''
        params:
          - src_tokens: (batch_size, src_len)
          - prev_tgt_tokens: (batch_size, tgt_len)

        returns:
          - model_out: (batch_size, tgt_len, n_tgt_words)
        '''

        src_mask = src_tokens.eq(self.src_pdx)
        tgt_mask = prev_tgt_tokens.eq(self.tgt_pdx)

        encoder_out = self.encoder(src_tokens, src_mask)
        decoder_out = self.decoder(
            prev_tgt_tokens, encoder_out, src_mask, tgt_mask)
        model_out = self.out_vocab_proj(decoder_out)
        return model_out

    def _model_init(self, encoder_prenorm, decoder_prenorm, N, M):
        self.encoder.encoder_alpha = 0.81 * (N ** 4 * M) ** (1 /16)
        encoder_beta = 0.87 * (N ** 4 * M) ** (-1 /16)
        self.encoder.decoder_alpha = (3 * M) ** (1 / 4)
        decoder_beta = (12 * M) ** (-1 / 4)
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            if not encoder_prenorm and 'encoder' in name:
                if 'embedding' in name:
                    nn.init.normal_(p, mean=0, std=self.d_model ** -0.5)
                    nn.init.constant_(p[self.src_pdx], 0)
                elif p.dim() > 1 and ('ffn' in name or 'w_v' in name or 'out_vocab_proj' in name):
                    nn.init.xavier_normal_(p, gain=encoder_beta)
                elif p.dim() > 1 and ('w_q' in name or 'w_k' in name):
                    nn.init.xavier_normal_(p, gain=1)

            if not decoder_prenorm and 'decoder' in name:
                if 'embedding' in name:
                    nn.init.normal_(p, mean=0, std=self.d_model ** -0.5)
                    nn.init.constant_(p[self.tgt_pdx], 0)
                elif p.dim() > 1 and ('ffn' in name or 'w_v' in name or 'out_vocab_proj' in name):
                    nn.init.xavier_normal_(p, gain=decoder_beta)
                elif p.dim() > 1 and ('w_q' in name or 'w_k' in name):
                    nn.init.xavier_normal_(p, gain=1)


class Encoder(nn.Module):
    def __init__(self, n_src_words, src_pdx, n_head, d_model, d_ff,
                 n_layers, p_drop, max_seq_len, encoder_prenorm):
        super().__init__()
        self.d_model = d_model
        self.encoder_alpha = 1.0
        self.dropout = nn.Dropout(p=p_drop)
        self.encoder_prenorm = encoder_prenorm
        self.input_embedding = nn.Embedding(
            num_embeddings=n_src_words, embedding_dim=d_model, padding_idx=src_pdx)
        self.positional_encode = PositionalEncode(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, n_head, p_drop, encoder_prenorm) for _ in range(n_layers)])
        if self.encoder_prenorm:
            self.layer_norm = nn.LayerNorm(d_model)  # for memory

    def forward(self, src_tokens, src_mask, **kwargs):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens) * (self.d_model ** 0.5)
        x = self.dropout(self.positional_encode(src_embed))
        for layer in self.layers:
            x = layer(x, src_mask, alpha=self.encoder_alpha)
        if self.encoder_prenorm:
            x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, p_drop, encoder_prenorm):
        super().__init__()
        self.encoder_prenorm = encoder_prenorm
        self.dropout = nn.Dropout(p=p_drop)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.pos_wise_ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, x, src_mask, alpha=1.0):
        if self.encoder_prenorm:
            x_ln = self.layernorm1(x)
            x = x + self.dropout(self.self_attn(
                q=x_ln, k=x_ln, v=x_ln,
                mask=src_mask.unsqueeze(1).unsqueeze(1)))
            x_ln = self.layernorm2(x)
            x = x + self.dropout(self.pos_wise_ffn(x_ln))
        else:
            x = self.layernorm1(x * alpha + self.dropout(self.self_attn(
                q=x, k=x, v=x,
                mask=src_mask.unsqueeze(1).unsqueeze(1))))
            x = self.layernorm2(x * alpha + self.dropout(self.pos_wise_ffn(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, n_tgt_words, tgt_pdx, n_head, d_model, d_ff,
                 n_layers, p_drop, max_seq_len, decoder_prenorm):
        super().__init__()
        self.d_model = d_model
        self.decoder_alpha = 1.0
        self.dropout = nn.Dropout(p=p_drop)
        self.input_embedding = nn.Embedding(
            num_embeddings=n_tgt_words, embedding_dim=d_model, padding_idx=tgt_pdx)
        self.positional_encode = PositionalEncode(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, n_head, p_drop, decoder_prenorm) for _ in range(n_layers)])

    def forward(self, prev_tgt_tokens, encoder_out, src_mask, tgt_mask, **kwargs):
        # - tgt_embed: (batch_size, src_len, d_model)
        tgt_embed = self.input_embedding(prev_tgt_tokens) * (self.d_model ** 0.5)
        x = self.dropout(self.positional_encode(tgt_embed))
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask, alpha=self.decoder_alpha)
        # - x: (batch_size, tgt_len, d_model)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, p_drop, decoder_prenorm):
        super().__init__()
        self.decoder_prenorm = decoder_prenorm
        self.dropout = nn.Dropout(p=p_drop)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.masked_self_attn = MultiHeadAttention(d_model, n_head)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.context_attn = MultiHeadAttention(d_model, n_head)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.pos_wise_ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, x, memory, src_mask, tgt_mask, alpha=1.0):
        if self.decoder_prenorm:
            x_ln = self.layernorm1(x)
            x = x + self.dropout(self.masked_self_attn(
                q=x_ln, k=x_ln, v=x_ln,
                mask=self._add_subsequent_mask(tgt_mask)))
            x_ln = self.layernorm2(x)
            x = x + self.dropout(self.context_attn(
                q=x_ln, k=memory, v=memory,
                mask=src_mask.unsqueeze(1).unsqueeze(1)))
            x_ln = self.layernorm3(x)
            x = x + self.dropout(self.pos_wise_ffn(x_ln))
        else:
            x = self.layernorm1(x * alpha + self.dropout(self.masked_self_attn(
                q=x, k=x, v=x,
                mask=self._add_subsequent_mask(tgt_mask))))
            x = self.layernorm2(x * alpha + self.dropout(self.context_attn(
                q=x, k=memory, v=memory,
                mask=src_mask.unsqueeze(1).unsqueeze(1))))
            x = self.layernorm3(x * alpha + self.dropout(self.pos_wise_ffn(x)))
        return x

    def _add_subsequent_mask(self, padding_mask):
        if padding_mask == None:
            return None
        # - padding_mask: (batch_size, seq_len)
        seq_len = padding_mask.size(1)
        subsequent_mask = torch.ones((seq_len, seq_len),
                                     device=padding_mask.device).triu(diagonal=1).bool()
        # - return: (batch_size, 1, seq_len, seq_len)
        return padding_mask.unsqueeze(1).unsqueeze(1) | subsequent_mask


class MultiHeadAttention(nn.Module):
    # - src_embed_dim = d_model
    def __init__(self, d_model, n_head):
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
    def __init__(self, d_model, d_ff, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class PositionalEncode(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.pos_encode = self._get_pos_encode(max_seq_len, d_model)

    def forward(self, x):
        # - x: (batch_size, seq_len, d_model)
        return x + self.pos_encode[:x.size(1), :].unsqueeze(0).to(x.device)

    def _get_pos_encode(self, max_seq_len, d_model):
        pos_encode = torch.tensor([[pos / 10000 ** (2 * (i // 2) / d_model) for i in range(d_model)]
                                   for pos in range(max_seq_len)], requires_grad=False)
        pos_encode[:, 0::2] = torch.sin(pos_encode[:, 0::2])
        pos_encode[:, 1::2] = torch.cos(pos_encode[:, 1::2])
        # - pos_encode: (seq_len, d_model)
        return pos_encode