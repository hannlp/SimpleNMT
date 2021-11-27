import os
import time
import dill
import logging
import torch
from tqdm import tqdm
from models import build_model
from torchtext.legacy import datasets
from data.dataloader import SortedIterator, batch_size_fn
from data.constants import Constants
from data.utils import prepare_batch
from .utils import de_numericalize
from .algorithms import beam_search, greedy_search


class Translator(object):
    def __init__(self, args):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_cuda = torch.cuda.is_available()
        self.model = self._load_model(ckpt_save_path=args.save_path, suffix=args.ckpt_suffix)
        self.model.eval()

        self.dl = torch.load('{}/{}-{}.dl'.format(
            args.save_path, args.src, args.tgt), pickle_module=dill)
        self.src_pdx = self.dl.src_pdx
        self.tgt_pdx = self.dl.tgt_pdx
        self.bos = self.dl.TGT.vocab.stoi[Constants.START]
        self.eos = self.dl.TGT.vocab.stoi[Constants.END]
        self.beam_size = args.beam_size
        self.length_penalty = args.length_penalty
        self.max_seq_len = args.max_seq_len

    def _load_model(self, ckpt_save_path, suffix):
        '''
        checkpoint(dict):
            - epoch(int)
            - model(dict): model.state_dict()
            - settings(NameSpace): train_args
        '''

        ckpt_path = '{}/checkpoint_{}.pt'.format(ckpt_save_path, suffix)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        checkpoint['settings'].use_cuda = self.use_cuda

        model_params = checkpoint['model']
        model_params = {key.replace("module.", ""): value for key, value in model_params.items()}

        model = build_model(checkpoint['settings'], training=False)
        model.load_state_dict(model_params)
        model.to(self.device)
        return model

    def generate(self, src, tgt, data_path, result_save_path, batch_size=4096, quiet=False):
        exts=('.' + src, '.' + tgt)
        test_path = data_path + '/test' if os.path.isdir(data_path) else data_path
        test = datasets.TranslationDataset(path=test_path, exts=exts, 
                    fields=(('src', self.dl.SRC), ('trg', self.dl.TGT)))

        # test_iter = Iterator(test, batch_size=batch_size, sort_key=None, device=None, batch_size_fn=None,
        #                      train=False, repeat=False, shuffle=None, sort=None, sort_within_batch=None
        # )
        test_iter = SortedIterator(test, batch_size=batch_size, device=None, repeat=False,
                               sort_key=lambda x: (len(x.src), len(x.trg)),
                               batch_size_fn=batch_size_fn, train=False, shuffle=True)
        
        result_path = result_save_path + '/result.txt'
        start_time = time.time()
        with open(result_path, 'w', encoding='utf8') as f, torch.no_grad(): 
            test_iter = tqdm(test_iter) if quiet else test_iter
            for batch in test_iter:
                src_tokens, _, tgt_tokens = prepare_batch(
                    batch, use_cuda=self.use_cuda)
                if self.beam_size > 0:
                    pred_tokens, _ = beam_search(model=self.model, src_tokens=src_tokens,
                            beam_size=self.beam_size, length_penalty=self.length_penalty,
                            max_seq_len=self.max_seq_len, bos=self.bos,
                            eos=self.eos, src_pdx=self.src_pdx, tgt_pdx=self.tgt_pdx)
                else:
                    pred_tokens = greedy_search(model=self.model, src_tokens=src_tokens,
                            max_seq_len=self.max_seq_len, bos=self.bos,
                            eos=self.eos, src_pdx=self.src_pdx, tgt_pdx=self.tgt_pdx)

                src_sentences = de_numericalize(self.dl.SRC.vocab, src_tokens)
                tgt_sentences = de_numericalize(self.dl.TGT.vocab, tgt_tokens)
                pred_sentences = de_numericalize(self.dl.TGT.vocab, pred_tokens)

                for src_words, tgt_words, pred_words in zip(src_sentences, tgt_sentences, pred_sentences):
                    content = '-S\t{}\n-T\t{}\n-P\t{}\n\n'.format(
                        ' '.join(src_words), ' '.join(tgt_words), ' '.join(pred_words))            
                    f.write(content)
                    if not quiet:
                        print(content)

        print('Successful. Generate time: {:.1f} min, the result has saved at {}'
                .format((time.time() - start_time) / 60, result_path))
    
    def translate(self, sentence, src, tgt, precise=False, bpe=None):
        if not precise:
            import jieba
            jieba.setLogLevel(logging.INFO)
            word_list = [w for w in list(jieba.cut(sentence)) if w.strip()]
        else:
            from .pipeline import input_pipeline
            word_list = input_pipeline(sentence, lang=src, bpe=bpe)

        with torch.no_grad():
            src_tokens = self.dl.SRC.numericalize([word_list], self.device) # (1, src_len)
            if self.beam_size > 0:
                pred_tokens, _ = beam_search(model=self.model, src_tokens=src_tokens,
                        beam_size=self.beam_size, length_penalty=self.length_penalty,
                        max_seq_len=self.max_seq_len, bos=self.bos,
                        eos=self.eos, src_pdx=self.src_pdx, tgt_pdx=self.tgt_pdx)
            else:
                pred_tokens = greedy_search(model=self.model, src_tokens=src_tokens,
                        max_seq_len=self.max_seq_len, bos=self.bos,
                        eos=self.eos, src_pdx=self.src_pdx, tgt_pdx=self.tgt_pdx)
            translated = de_numericalize(self.dl.TGT.vocab, pred_tokens)[0]
        
        if not precise:
            print(' '.join(translated), end="\n")
        else:
            from .pipeline import output_pipeline
            print(output_pipeline(translated, lang=tgt))