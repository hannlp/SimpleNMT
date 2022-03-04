import torch
import torch.nn as nn
import torch.nn.functional as F


class FLASH(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, src_pdx=-1, tgt_pdx=-1,
                 d_model=512, d_ff=2048, n_head=8, n_encoder_layers=6,
                 n_decoder_layers=6, p_drop=0.1, max_seq_len=512):

        super().__init__()
        self.d_model = d_model
        self.src_pdx, self.tgt_pdx = src_pdx, tgt_pdx

        self.encoder = Encoder(n_src_words, src_pdx=src_pdx, n_head=n_head,
                               d_model=d_model, d_ff=d_ff, n_layers=n_encoder_layers,
                               p_drop=p_drop, max_seq_len=max_seq_len)

        self.decoder = Decoder(n_tgt_words, tgt_pdx=tgt_pdx, n_head=n_head,
                               d_model=d_model, d_ff=d_ff, n_layers=n_decoder_layers,
                               p_drop=p_drop, max_seq_len=max_seq_len)

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
        tgt_mask = prev_tgt_tokens.eq(self.tgt_pdx)

        encoder_out = self.encoder(src_tokens, src_mask)
        decoder_out = self.decoder(
            prev_tgt_tokens, encoder_out, src_mask, tgt_mask)
        model_out = self.out_vocab_proj(decoder_out)
        return model_out

    def _model_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Encoder(nn.Module):
    def __init__(self, n_src_words, src_pdx, n_head, d_model, d_ff,
                 n_layers, p_drop, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=p_drop)
        self.input_embedding = nn.Embedding(
            num_embeddings=n_src_words, embedding_dim=d_model, padding_idx=src_pdx)
        self.positional_encode = PositionalEncode(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [GAU(d_model) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)  # for memory

    def forward(self, src_tokens, src_mask, **kwargs):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens) * (self.d_model ** 0.5)
        x = self.dropout(self.positional_encode(src_embed))
        for layer in self.layers:
            x = layer(x, src_mask)
        encoder_out = self.layer_norm(x)
        return encoder_out


class Decoder(nn.Module):
    def __init__(self, n_tgt_words, tgt_pdx, n_head, d_model, d_ff,
                 n_layers, p_drop, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=p_drop)
        self.input_embedding = nn.Embedding(
            num_embeddings=n_tgt_words, embedding_dim=d_model, padding_idx=tgt_pdx)
        self.positional_encode = PositionalEncode(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, n_head, p_drop) for _ in range(n_layers)])

    def forward(self, prev_tgt_tokens, encoder_out, src_mask, tgt_mask, **kwargs):
        # - tgt_embed: (batch_size, src_len, d_model)
        tgt_embed = self.input_embedding(prev_tgt_tokens) * (self.d_model ** 0.5)
        x = self.dropout(self.positional_encode(tgt_embed))
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        # - decoder_out: (batch_size, tgt_len, n_tgt_words)
        decoder_out = x
        return decoder_out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, p_drop):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)
        self.sublayer1_prenorm = nn.LayerNorm(d_model)
        self.masked_self_attn = MultiHeadAttention(d_model, n_head)
        self.sublayer2_prenorm = nn.LayerNorm(d_model)
        self.context_attn = MultiHeadAttention(d_model, n_head)
        self.sublayer3_prenorm = nn.LayerNorm(d_model)
        self.pos_wise_ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, x, memory, src_mask, tgt_mask):
        res, x_ln = x, self.sublayer1_prenorm(x)
        x = res + self.dropout(self.masked_self_attn(
            q=x_ln, k=x_ln, v=x_ln, mask=self._add_subsequent_mask(tgt_mask)))
        res, x_ln = x, self.sublayer2_prenorm(x)
        x = res + self.dropout(self.context_attn(
            q=x_ln, k=memory, v=memory, mask=src_mask.unsqueeze(1).unsqueeze(1)))
        res, x_ln = x, self.sublayer3_prenorm(x)
        x = res + self.dropout(self.pos_wise_ffn(x_ln))
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


class GAU(nn.Module):
    def __init__(self, d_model, s=128, e=1536):
        super().__init__()
        #self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.w_z = nn.Linear(d_model, s) # for q, k
        self.w_u = nn.Linear(d_model, e)
        self.w_v = nn.Linear(d_model, e)
        self.w_o = nn.Linear(e, d_model)

    def scale_offset(self, x):
        gamma = torch.empty(x.shape[1:], device=x.device) # x.shape[-1:] in the paper
        beta = torch.empty(x.shape[1:], device=x.device)
        torch.nn.init.normal_(gamma, std=0.02)
        torch.nn.init.normal_(gamma, mean=1e-5)
        return x * gamma + beta

    def attn(self, x, v, mask=None):
        z = self.w_z(x)
        q, k = self.scale_offset(z), self.scale_offset(z)  # B x T x s
        qk = torch.matmul(q, torch.transpose(k, 1, 2))
        a = F.relu(qk) ** 2  # B x T x T

        if mask != None:
            a.masked_fill_(mask.unsqueeze(1), 0)

        return torch.matmul(a, v)

    def forward(self, x, mask=None):
        shortcut, x = x, self.norm(x)
        u, v = self.w_u(x), self.w_v(x)
        x = u * self.attn(x, v, mask=mask)
        return self.w_o(x) + shortcut

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

if __name__ == '__main__':
    B, Ts, Tt ,D = 2, 3, 4, 5
    # src_mask = torch.randint(0, 2, (B, Ts)).bool()
    # gau = GAU(D)
    # x = torch.randn(B, Ts, D)
    # y = gau(x, src_mask)
    # print(x, y)

    n_src_words = 20
    n_tgt_words = 25
    batch_size = 5
    src_len = 64
    tgt_len = 60

    model = FLASH(n_src_words=n_src_words, n_tgt_words=n_tgt_words)
    print(model)

    src_tokens = torch.randint(n_src_words, (batch_size, src_len))
    prev_output_tokens = torch.randint(n_tgt_words, (batch_size, tgt_len))

    out = model(src_tokens, prev_output_tokens)
    print(out.shape)