import torch
import torch.nn as nn
import argparse
from data.dataloader import DataLoader
from train.trainer import Trainer
from models import build_model
from train.loss import LabelSmoothingLoss

def parse():
    # The arguments for DataLoader and Trainer
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", help="the source language", type=str, default="zh")
    parser.add_argument("-tgt", help="the target language", type=str, default="en")
    parser.add_argument("-data_path", help="the path prefix of whole data which will to be split", type=str, default="")
    parser.add_argument("-train_path", help="the train corpus path prefix", type=str, default="")
    parser.add_argument("-valid_path", help="the valid corpus path prefix", type=str, default="")
    parser.add_argument("-dl_path", help="the dataloader save path", type=str, default="./temp.dl")
    parser.add_argument("-ckpt_path", help="the checkpoint save path", type=str, default=".")
    parser.add_argument("-batch_size", type=int, default=3200)
    parser.add_argument("-warmup_steps", help="warmup steps of learning rate update", type=int, default=4000)
    parser.add_argument("-n_epochs", type=int, default=30)

    # The arguments for Transformer
    parser.add_argument("-model", help="model name", type=str, default='Transformer')
    parser.add_argument("-d_model", help="dimension of the model", type=int, default=512)
    parser.add_argument("-n_layers", type=int, default=6)
    parser.add_argument("-n_head", help="number of heads in multihead-attention", type=int, default=8)
    parser.add_argument("-p_drop", help="probability of dropout", type=float, default=0.1)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-betas", type=float, nargs="+", default=(0.9, 0.98))
    parser.add_argument("-max_seq_len", type=int, default=2048)
    parser.add_argument("-share_vocab", help="share src tgt embeddings and share decoder embeddings", action="store_true")
    #parser.add_argument("-share_decoder_embeddings", action="store_true")

    # The arguments for RNNs
    parser.add_argument("-bidirectional", help="use bidirectional rnns", action="store_true")
    parser.add_argument("-attn_type", help="type of attention from luong's paper", type=str, default='general')
    parser.add_argument("-rnn_type", help="type of RNNs", type=str, default='gru')
    
    parser.add_argument("-label_smoothing", type=float, default=0.1)
    args = parser.parse_args()
    return args

def main():
    use_cuda = torch.cuda.is_available()
    args = parse()
    dl = DataLoader()
    train_iter, valid_iter = dl.load_translation(
        exts=('.' + args.src, '.' + args.tgt), # default: ('.zh', '.en')
        data_path=args.data_path,
        train_path=args.train_path,
        valid_path=args.valid_path,      
        batch_size=args.batch_size, 
        dl_save_path=args.dl_path,
        share_vocab=args.share_vocab
        )
    
    args.n_src_words, args.n_tgt_words = len(dl.SRC.vocab), len(dl.TGT.vocab)
    args.src_pdx, args.tgt_pdx = dl.src_padding_index, dl.tgt_padding_index
    print(args)
  
    model = build_model(args, use_cuda=use_cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=args.betas, eps=1e-9)
    if args.label_smoothing:
        criterion = LabelSmoothingLoss(args.label_smoothing, ignore_index=args.tgt_pdx, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.tgt_pdx, reduction='mean')
    trainer = Trainer(args, model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      use_cuda=use_cuda
                      )
    trainer.train(train_iter, valid_iter, n_epochs=args.n_epochs,
                  save_path=args.ckpt_path)

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    main()
