import os
import torch
import argparse
from data.dataloader import DataLoader
from train import build_criterion, build_optimizer, get_logger, set_seed
from train.trainer import Trainer
from models import build_model, count_parameters

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
    parser.add_argument("-log_interval", help="the steps interval of train log", type=int, default=100)
    parser.add_argument("-keep_last_ckpts", help="the num of saving last checkpoints", type=int, default=5)
    parser.add_argument("-optim", help="the optimizer for training", type=str, default="noam")
    parser.add_argument("-seed", help="for reproducibility", type=int, default=1314)
    parser.add_argument("-split_ratio", help="the ratio of train set, and the reset is valid set", type=float, default=0.95)
    parser.add_argument("-patience", help="early stop after patience epochs of nothing better checkpoints", type=int, default=10)

    # The arguments for all models
    parser.add_argument("-model", help="model name", type=str, default='Transformer')
    parser.add_argument("-d_model", help="dimension of the model", type=int, default=512)
    parser.add_argument("-n_layers", type=int, default=6)
    parser.add_argument("-n_encoder_layers", type=int, default=6)
    parser.add_argument("-n_decoder_layers", type=int, default=6)
    parser.add_argument("-share_vocab", help="share src tgt embeddings and share decoder embeddings", action="store_true")
    #parser.add_argument("-share_decoder_embeddings", action="store_true")
    parser.add_argument("-p_drop", help="probability of dropout", type=float, default=0.1)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-lr_scale", help="a scale for learning rate", type=float, default=1.0)
    parser.add_argument("-betas", type=float, nargs="+", default=(0.9, 0.98))

    # The arguments for Transformer
    parser.add_argument("-n_head", help="number of heads in multihead-attention", type=int, default=8)
    parser.add_argument("-d_ff", type=int, default=2048)
    parser.add_argument("-label_smoothing", type=float, default=0.1)
    parser.add_argument("-warmup_steps", help="warmup steps of learning rate update", type=int, default=4000)
    parser.add_argument("-encoder_prenorm", action="store_true")
    parser.add_argument("-decoder_prenorm", action="store_true")

    # The arguments for RNNs
    parser.add_argument("-bidirectional", help="use bidirectional rnns", action="store_true")
    parser.add_argument("-attn_type", help="type of attention from luong's paper", type=str, default='general')
    parser.add_argument("-rnn_type", help="type of RNNs", type=str, default='gru')
    args = parser.parse_args()
    return args

def main():
    args = parse()

    # For reproducibility
    set_seed(args.seed, deterministic=False)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger = get_logger(args)

    # Build dataloader and load translation dataset
    dl = DataLoader()
    train_iter, valid_iter = dl.load_translation(
            src=args.src, tgt=args.tgt, batch_size=args.batch_size,
            data_path=args.data_path, dl_save_path=args.save_path,
            share_vocab=args.share_vocab, split_ratio=args.split_ratio,
            logger=logger)
    dl.write_vocab(args.save_path)

    args.n_src_words, args.n_tgt_words = len(dl.SRC.vocab), len(dl.TGT.vocab)
    args.src_pdx, args.tgt_pdx = dl.src_pdx, dl.tgt_pdx
    args.use_cuda = torch.cuda.is_available()
    logger.info(args)

    # Build trainer and start training
    model = build_model(args)
    count_parameters(model, logger)
    optimizer = build_optimizer(args, model)
    criterion = build_criterion(args)
    trainer = Trainer(args=args, model=model, optimizer=optimizer,
                      criterion=criterion, lr_scale=args.lr_scale, logger=logger)

    trainer.train(train_iter, valid_iter, n_epochs=args.n_epochs, patience=args.patience,
                  log_interval=args.log_interval, ckpt_save_path=args.save_path)

if __name__ == '__main__':
    main()
