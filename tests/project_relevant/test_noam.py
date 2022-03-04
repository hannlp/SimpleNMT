import sys
import os
print(os.path, os.getcwd())
sys.path.append(os.getcwd())
import torch
from simplenmt.models.transformer_legacy import Transformer
from simplenmt.train.optim import Noam

n_src_words = 20
n_tgt_words = 25
batch_size = 5
src_len = 64
tgt_len = 60

model = Transformer(n_src_words=n_src_words, n_tgt_words=n_tgt_words)
optimizer = Noam(optimizer=torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9), lr_scale=2, d_model=512, warmup_steps=1000)
print(optimizer.param_groups[0]['lr'])

for i in range(500):
    optimizer.step()
    if i % 50 == 0:
        print(optimizer.param_groups[0]['lr'])

# adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
# print(adam.state_dict)

print(optimizer.state_dict)