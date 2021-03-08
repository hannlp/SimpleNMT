import torch
import torch.nn as nn
import argparse
import dill
from data.dataloader import DataLoader
from train.trainer import Trainer
from models.transformer import Transformer
from utils.builder import build_model

CUDA_OK = torch.cuda.is_available()

def parse():
    # The arguments for Trainer
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--data_path", type=str, default=".")
    parser.add_argument("-dl_path", "--dataloader_save_path", help="the dataloader save path", type=str, default=".")
    parser.add_argument("-ckpt_path", "--checkpoint_save_path", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=3200)
    parser.add_argument("--warmup_steps", help="warmup steps of learning rate update", type=int, default=4000)
    parser.add_argument("--n_epochs", type=int, default=20)

    # The arguments for Transformer
    parser.add_argument("--d_model", help="dimension of the model", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", help="heads number of mutihead-attention", type=int, default=8)
    parser.add_argument("--p_drop", help="probability of dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.98))
    
    args = parser.parse_args()
    args.constants = {'PAD': '<pad>', 'START': '<sos>',
                 'END': '<eos>', 'UNK': '<unk>'}
    return args

def main():
    args = parse()
    dl = DataLoader(**args.constants)
    print(args)
    train_iter, valid_iter = dl.load_translation(
        path=args.path, exts=('.en', '.zh'), batch_size=4800, dl_save_path=args.dl_path, device="cuda:0")
        
    args.n_src_words, args.n_tgt_words = len(dl.SRC.vocab), len(dl.TGT.vocab)
    args.src_pdx, args.tgt_pdx = dl.SRC.vocab.stoi[dl.PAD], dl.TGT.vocab.stoi[dl.PAD]
    print(args)

    model = build_model(args, Transformer, CUDA_OK)
    trainer = Trainer(model=model,
                      optimizer=torch.optim.Adam(
                          model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9),
                      criterion=nn.CrossEntropyLoss(
                          ignore_index=args['tgt_pdx'], reduction='mean'),
                      warmup_steps=4000, d_model=args['d_model'])
    trainer.train(train_iter, valid_iter, n_epochs=8,
                  save_path='/home/hanyuchen/NMT/checkpoints')

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    main()
