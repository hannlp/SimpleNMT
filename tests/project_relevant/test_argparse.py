import argparse

# The arguments for DataLoader and Trainer
parser = argparse.ArgumentParser()
parser.add_argument("-src", help="the source language", type=str, default="zh")
parser.add_argument("-tgt", help="the target language", type=str, default="en")
parser.add_argument("-data_path", help="the path prefix of whole data which will to be split", type=str, default="")
parser.add_argument("-train_path", help="the train corpus path prefix", type=str, default="")
parser.add_argument("-valid_path", help="the valid corpus path prefix", type=str, default="")
parser.add_argument("-dl_path", help="the dataloader save path", type=str, default="")
parser.add_argument("-ckpt_path", help="the checkpoint save path", type=str, default="")
parser.add_argument("-batch_size", type=int, default=3200)
parser.add_argument("-warmup_steps", help="warmup steps of learning rate update", type=int, default=4000)
parser.add_argument("-n_epochs", type=int, default=20)

# The arguments for Transformer
parser.add_argument("-model", help="model name", type=str, default='Transformer')
parser.add_argument("-d_model", help="dimension of the model", type=int, default=512)
parser.add_argument("-n_layers", type=int, default=6)
parser.add_argument("-n_head", help="number of heads in multihead-attention", type=int, default=8)
parser.add_argument("-p_drop", help="probability of dropout", type=float, default=0.1)
parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("-betas", type=float, nargs="+", default=(0.9, 0.98))
parser.add_argument("-max_seq_len", type=int, default=512)
parser.add_argument("-share_vocab", help="share src tgt embeddings and share decoder embeddings", action="store_true")

# The parser for Translator


args = parser.parse_args()

# define arguments after parse
args.a_int = 1
args.a_str = 'this is a string.'
args.a_float = 3.1415926

print(args)