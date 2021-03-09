import torch
import torch.nn as nn
import torch.nn.functional as F
import dill
from typing import Union
from utils.builder import build_model
from data.utils import prepare_batch


class Translator(object):
    def __init__(self, args):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.dl = torch.load(args.dl_path, pickle_module=dill)
        self.src_pdx = self.dl.src_padding_index
        self.tgt_pdx = self.dl.tgt_padding_index
        self.tgt_sos_idx = self.dl.TGT.vocab.stoi[self.dl.START]
        self.tgt_eos_idx = self.dl.TGT.vocab.stoi[self.dl.END]
        self.src_stoi = self.dl.SRC.vocab.stoi
        self.tgt_itos = self.dl.TGT.vocab.itos

        self.max_length = args.max_length

        self.model = self._load_model(args)
        self.model.eval()

    def _load_model(self, args):
        '''
        checkpoint(dict):
            - epoch(int)
            - model(dict): model.state_dict()
            - settings(NameSpace): train_args
        '''

        checkpoint = torch.load(
            '{}/checkpoint_best.pt'.format(args.ckpt_path), 
            map_location=self.device
            )

        model = build_model(checkpoint['settings'], cuda_ok=torch.cuda.is_available())
        model.load_state_dict(checkpoint['model'])
        if hasattr(model, 'module'):
            model = self.model.module
        model.to(self.device)
        return model

    def generate(self, data_iter, CUDA_OK):
        abatch = next(iter(data_iter))
        src_tokens, prev_tgt_tokens, tgt_tokens = prepare_batch(
            abatch, CUDA_OK)
        with torch.no_grad():
            out_tokens = torch.argmax(
                nn.functional.softmax(self.model(src_tokens, prev_tgt_tokens), dim=-1), dim=-1)

        def show_src_tgt_out(src, tgt, out):
            batch_size = out.size(0)
            for b in range(batch_size):
                print('\n|src: ', end=" ")
                for i in range(src.size(1)):
                    print(self.SRC_VOCAB.itos[src[b, i]], end=' ')
                print('\n|gold: ', end=" ")
                for i in range(tgt.size(1)):
                    print(self.TGT_VOCAB.itos[tgt[b, i]], end='')
                print('\n|out: ', end=" ")
                for i in range(out.size(1)):
                    print(self.TGT_VOCAB.itos[out[b, i]], end='')
                print()
        show_src_tgt_out(src_tokens, tgt_tokens, out_tokens)

    def _greedy_search(self, sentence):
        src_tokens = torch.tensor([self.src_stoi[s]
                                   for s in sentence.split()]).unsqueeze(0).to(self.device)
        src_mask = (src_tokens != self.src_pdx).bool().to(self.device)
        encoder_out = self.model.encoder(src_tokens, src_mask)
        prev_tgt_tokens = torch.tensor([self.tgt_sos_idx]).unsqueeze(0).to(self.device)  # <sos>
        tgt_mask = (prev_tgt_tokens != self.tgt_pdx).bool().to(self.device)
        decoder_out = F.softmax(self.model.decoder(
            prev_tgt_tokens, encoder_out, src_mask, tgt_mask)[0], dim=-1)
        topk_probs, topk_idx = decoder_out[:, -1, :].topk(1)
        for step in range(2, self.max_length):
            new_word = torch.tensor(topk_idx[:, 0]).unsqueeze(0)
            if new_word == self.tgt_eos_idx:
                break
            prev_tgt_tokens = torch.cat(
                (prev_tgt_tokens, new_word), dim=1)  # (1, step)
            tgt_mask = (prev_tgt_tokens != self.tgt_pdx).bool().to(self.device)
            decoder_out = F.softmax(self.model.decoder(
                prev_tgt_tokens, encoder_out, src_mask, tgt_mask)[0], dim=-1)
            # print(decoder_out.shape) # (1, 1(step), tgt_vocab_size)
            topk_probs, topk_idx = decoder_out[:, -1, :].topk(1)
        return ' '.join([self.tgt_itos[w_id] for w_id in list(prev_tgt_tokens.squeeze().detach()[1:])])

    def translate(self, sentence: str, beam_size=8):
        with torch.no_grad():
            if beam_size == 1:
                return self._greedy_search(sentence)
            else:
                print('Not implementation. ')
