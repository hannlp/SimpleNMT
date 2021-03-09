import torch
import torch.nn as nn
import argparse
from data.dataloader import DataLoader
from train.trainer import Trainer
from utils.builder import build_model

def parse():
    # The arguments for Trainer
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", help="the source language", type=str, default="zh")
    parser.add_argument("-tgt", help="the target language", type=str, default="en")
    parser.add_argument("-data_path", help="the train corpus path", type=str, default="./")
    parser.add_argument("-dl_path", help="the dataloader save path", type=str, default="./")
    parser.add_argument("-ckpt_path", help="the checkpoint save path", type=str, default="./")
    parser.add_argument("-batch_size", type=int, default=3200)
    parser.add_argument("-warmup_steps", help="warmup steps of learning rate update", type=int, default=4000)
    parser.add_argument("-n_epochs", type=int, default=20)

    # The arguments for Transformer
    parser.add_argument("-model", help="model name", type=str, default='Transformer')
    parser.add_argument("-d_model", help="dimension of the model", type=int, default=512)
    parser.add_argument("-n_layer", type=int, default=6)
    parser.add_argument("-n_head", help="number of heads in multihead-attention", type=int, default=8)
    parser.add_argument("-p_drop", help="probability of dropout", type=float, default=0.1)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-betas", type=float, nargs="+", default=(0.9, 0.98))
    
    args = parser.parse_args()
    return args

def main():

    args = parse()
    dl = DataLoader()
    train_iter, valid_iter = dl.load_translation(
        data_path=args.data_path, 
        exts=('.' + args.src, '.' + args.tgt), # ('.zh', '.en')
        batch_size=args.batch_size, 
        dl_save_path=args.dl_path
        )
    
    args.n_src_words, args.n_tgt_words = len(dl.SRC.vocab), len(dl.TGT.vocab)
    args.src_pdx, args.tgt_pdx = dl.src_padding_index, dl.tgt_padding_index
    print(args)
  
    model = build_model(args, cuda_ok=torch.cuda.is_available())
    trainer = Trainer(args, model=model,
                      optimizer=torch.optim.Adam(
                          model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9),
                      criterion=nn.CrossEntropyLoss(
                          ignore_index=args.tgt_pdx, reduction='mean')
                          )
    trainer.train(train_iter, valid_iter, n_epochs=args.n_epochs,
                  save_path=args.ckpt_path)

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    main()
