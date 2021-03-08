import torch
import torch.nn as nn
import dill
from .data.dataloader import DataLoader
from .models.transformer import Transformer
from .utils.builder import build_model

CUDA_OK = torch.cuda.is_available()


def main():

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    constants = {'PAD': '<pad>', 'START': '<sos>',
                 'END': '<eos>', 'UNK': '<unk>'}
    en_zh = DataLoader(**constants)

    dl_save_path = '/home/hanyuchen/NMT/en_zh.dl'
    train_iter, valid_iter = en_zh.load_translation(
        path='~/NMT/clean', exts=('.en', '.zh'), batch_size=4800, dl_save_path=dl_save_path, device="cuda:0")

    args = {'n_src_words': len(en_zh.SRC.vocab),
            'n_tgt_words': len(en_zh.TGT.vocab),
            'src_pdx': en_zh.SRC.vocab.stoi[en_zh.PAD],
            'tgt_pdx': en_zh.TGT.vocab.stoi[en_zh.PAD],
            'd_model': 256, 'n_layer': 3,
            'n_head': 8, 'p_drop': 0.1}

    model = build_model(args, Transformer, CUDA_OK)
    trainer = Trainer(model=model,
                      optimizer=torch.optim.Adam(
                          model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9),
                      criterion=nn.CrossEntropyLoss(
                          ignore_index=args['tgt_pdx'], reduction='mean'),
                      warmup_steps=4000, d_model=args['d_model'])
    trainer.train(train_iter, valid_iter, n_epochs=8,
                  save_path='/home/hanyuchen/NMT/checkpoints')
    EN_ZH = torch.load(dl_save_path, pickle_module=dill)
    translator = Translator(args, Transformer, EN_ZH,
                            load_path='/home/hanyuchen/NMT/checkpoints')
    translator.generate(valid_iter)

if __name__ == '__main__':
    main()
