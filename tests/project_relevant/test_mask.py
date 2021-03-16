import torch

# padding mask
src_tokens = torch.randint(0, 5, (5, 10))
src_pdx = 0

print(src_tokens)
print(src_tokens != src_pdx)
print(src_tokens.eq(src_pdx))

# sequence mask
seq_len = 5

old_sequence_mask = torch.tril(torch.ones((seq_len, seq_len))).bool()
sequence_mask = torch.ones((seq_len, seq_len)).triu(diagonal=1).bool()
print(old_sequence_mask, sequence_mask)

a = torch.tensor([True, True, False, False])
b = torch.tensor([True, False, True, False])
print(a | b , a & b)