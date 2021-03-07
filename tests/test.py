import torch
print(__file__)
print(__name__)
print(__package__)
from ..simplenmt.models.transformer import Transformer

model = Transformer(n_src_words=20, n_tgt_words=25)
print(model)

# 最多20个单词， 一个batch5句，src每句最长64, tgt每句最长60
src_tokens = torch.randint(20, (5, 64))
prev_output_tokens = torch.randint(25, (5, 60))


out = model(src_tokens, prev_output_tokens)
print(out.shape)
