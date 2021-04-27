import os
import torch
import argparse
from data.dataloader import DataLoader
from train.utils import get_logger
from train.trainer import Trainer
from models import build_model, count_parameters
from train import build_criterion

def parse():
    parser = argparse.ArgumentParser()

    # The arguments for DataLoader and Trainer
    parser.add_argument("-src", help="the source language", type=str, default="zh")
    parser.add_argument("-tgt", help="the target language", type=str, default="en")
    parser.add_argument("-data_path", help="the path prefix of whole data which will to be split, or path of train and valid", type=str, default=".")
    parser.add_argument("-save_path", help="the path to save checkpoint, dataloader and log", type=str, default=".")
    parser.add_argument("-batch_size", type=int, default=4096)
    parser.add_argument("-max_seq_len", type=int, default=2048)
    parser.add_argument("-n_epochs", type=int, default=40)
    
    # The arguments for all models
    parser.add_argument("-model", help="model name", type=str, default='Transformer')
    parser.add_argument("-d_model", help="dimension of the model", type=int, default=512)
    parser.add_argument("-n_layers", type=int, default=6)
    parser.add_argument("-share_vocab", help="share src tgt embeddings and share decoder embeddings", action="store_true")
    #parser.add_argument("-share_decoder_embeddings", action="store_true")
    parser.add_argument("-p_drop", help="probability of dropout", type=float, default=0.1)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-betas", type=float, nargs="+", default=(0.9, 0.98))
    
    # The arguments for Transformer
    parser.add_argument("-n_head", help="number of heads in multihead-attention", type=int, default=8)
    parser.add_argument("-label_smoothing", type=float, default=0.1)
    parser.add_argument("-warmup_steps", help="warmup steps of learning rate update", type=int, default=4000)

    # The arguments for RNNs
    parser.add_argument("-bidirectional", help="use bidirectional rnns", action="store_true")
    parser.add_argument("-attn_type", help="type of attention from luong's paper", type=str, default='general')
    parser.add_argument("-rnn_type", help="type of RNNs", type=str, default='gru')
    args = parser.parse_args()
    return args

def main():
    args = parse()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger = get_logger(args)
    dl = DataLoader()
    train_iter, valid_iter = dl.load_translation(
            src=args.src, tgt=args.tgt,
            data_path=args.data_path,
            batch_size=args.batch_size,
            dl_save_path=args.save_path,
            share_vocab=args.share_vocab,
            logger=logger)
    
    args.n_src_words, args.n_tgt_words = len(dl.SRC.vocab), len(dl.TGT.vocab)
    args.src_pdx, args.tgt_pdx = dl.src_padding_index, dl.tgt_padding_index
    args.use_cuda = torch.cuda.is_available()
    logger.info(args)

    model = build_model(args)
    count_parameters(model, logger)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=args.betas, eps=1e-9)
    criterion = build_criterion(args)
    trainer = Trainer(args, model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      logger=logger)
    trainer.train(train_iter, valid_iter, n_epochs=args.n_epochs,
                  ckpt_save_path=args.save_path)

if __name__ == '__main__':
    main()
