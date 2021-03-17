import torch
import torch.nn as nn
import torch.nn.functional as F
import dill
import jieba
import logging
from torchtext import datasets
from utils.builder import build_model
from data.dataloader import MyIterator, batch_size_fn
from data.utils import prepare_batch
from data.constants import Constants
#from data.utils import prepare_batch


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

        self.max_seq_length = args.max_seq_length

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

        model = build_model(checkpoint['settings'], use_cuda=torch.cuda.is_available())
        model.load_state_dict(checkpoint['model'])
        if hasattr(model, 'module'):
            model = model.module
        model.to(self.device)
        return model

    # TODO: 实现批量生成pred
    def generate(self, test_path, exts, batch_size=3200):

        def de_numericalize(vocab, arr, 
                remove_constants={Constants.PAD, Constants.START, Constants.END}):
            arr = [[vocab.itos[x] for x in ex if vocab.itos[x] not in remove_constants] for ex in arr]
            return arr

        test = datasets.TranslationDataset(
            path=test_path, exts=exts, 
            fields=(('src', self.dl.SRC), ('trg', self.dl.TGT)))
        
        test_iter = MyIterator(test, batch_size=batch_size, device=None,
                                repeat=False, sort_key=lambda x:
                                (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True,
                                shuffle=True)

        print('Writing result to {} ...'.format(test_path + '.result'))
        with open(test_path + '.result', 'w', encoding='utf8') as f:
            with torch.no_grad():
                for _, batch in enumerate(test_iter, start=1):
                    src_tokens, _, tgt_tokens = prepare_batch(
                        batch, use_cuda=torch.cuda.is_available())

                    src_words_list = de_numericalize(self.dl.SRC.vocab, src_tokens)
                    tgt_words_list = de_numericalize(self.dl.TGT.vocab, tgt_tokens)
                    
                    pred_tokens = self.batch_greedy_search(src_tokens)
                    pred_words_list = de_numericalize(self.dl.TGT.vocab, pred_tokens) # 记得换成TGT

                    for src_words, tgt_words, pred_words in zip(src_words_list, tgt_words_list, pred_words_list):
                        f.write('-S: {}'.format(' '.join(src_words)) + '\n')
                        f.write('-T: {}'.format(' '.join(tgt_words)) + '\n')
                        f.write('-P: {}'.format(' '.join(pred_words)) + '\n\n')
  
    def batch_greedy_search(self, src_tokens):
        batch_size = src_tokens.size(0)
        src_mask = src_tokens.eq(self.src_pdx).to(self.device)
        encoder_out = self.model.encoder(src_tokens, src_mask)
        gen_seqs = torch.full((batch_size, 1), self.tgt_sos_idx).to(self.device) # prev_tgt_tokens
        tgt_mask = gen_seqs.eq(self.tgt_pdx).to(self.device)
        decoder_out = self.model.decoder(
            gen_seqs, encoder_out, src_mask, tgt_mask)
        out = F.softmax(self.model.out_vocab_proj(decoder_out), dim=-1)
        _, max_idxs = out[:, -1, :].topk(1)
        for step in range(2, 10):
            new_words = max_idxs.to(self.device)
            #TODO : stop rules
            # if new_word == self.tgt_eos_idx:
            #     break
            gen_seqs = torch.cat((gen_seqs, new_words), dim=1)  # (batch_size, step)
            tgt_mask = gen_seqs.eq(self.tgt_pdx).to(self.device)

            decoder_out = self.model.decoder(
                gen_seqs, encoder_out, src_mask, tgt_mask)
            out = F.softmax(self.model.out_vocab_proj(decoder_out), dim=-1)
            # print(out.shape) # (1, 1(step), tgt_vocab_size)
            _, max_idxs = out[:, -1, :].topk(1)
        
        #return ' '.join([self.tgt_itos[w_id] for w_id in list(prev_tgt_tokens.squeeze().detach()[1:])])
        return gen_seqs#src_tokens

    # TODO: 由于最后一层线性映射从decoder换到了transformer，所以这里面都需要调整
    def _greedy_search(self, word_list):
        src_tokens = torch.tensor([[self.src_stoi[s]
                                   for s in word_list]]).to(self.device)
        src_mask = src_tokens.eq(self.src_pdx).to(self.device)
        encoder_out = self.model.encoder(src_tokens, src_mask)

        prev_tgt_tokens = torch.tensor([[self.tgt_sos_idx]]).to(self.device)  # <sos>
        tgt_mask = prev_tgt_tokens.eq(self.tgt_pdx).to(self.device)
        decoder_out = self.model.decoder(
            prev_tgt_tokens, encoder_out, src_mask, tgt_mask)
        out = F.softmax(self.model.out_vocab_proj(decoder_out), dim=-1)

        _, max_idx = out[:, -1, :].topk(1)

        for step in range(2, self.max_seq_length):
            new_word = max_idx[:, 0].unsqueeze(0).to(self.device)
            if new_word == self.tgt_eos_idx:
                break
            prev_tgt_tokens = torch.cat(
                (prev_tgt_tokens, new_word), dim=1)  # (1, step)
            tgt_mask = prev_tgt_tokens.eq(self.tgt_pdx).to(self.device)

            decoder_out = self.model.decoder(
                prev_tgt_tokens, encoder_out, src_mask, tgt_mask)
            out = F.softmax(self.model.out_vocab_proj(decoder_out), dim=-1)
            # print(out.shape) # (1, 1(step), tgt_vocab_size)
            _, max_idx = out[:, -1, :].topk(1)
        
        return ' '.join([self.tgt_itos[w_id] for w_id in list(prev_tgt_tokens.squeeze().detach()[1:])])

    def _beam_search(self, word_list, beam_size=8):
        len_map = torch.arange(1, self.max_seq_len + 1, dtype=torch.long).unsqueeze(0).to(self.device)
        
        def decode(prev_tgt_tokens, encoder_out, src_mask):
            tgt_mask = (prev_tgt_tokens != self.tgt_pdx).to(self.device)
            decoder_out = F.softmax(self.model.decoder(prev_tgt_tokens, encoder_out, src_mask, tgt_mask), dim=-1)
            return decoder_out

        gen_seqs = torch.full((beam_size, 1), self.tgt_sos_idx).to(self.device)

        with torch.no_grad():
            src_tokens = torch.tensor([self.src_stoi[s] for s in word_list]).unsqueeze(0).to(self.device)
            src_mask = (src_tokens != self.src_pdx).to(self.device)

            encoder_out = self.model.encoder(src_tokens, src_mask)

            prev_tgt_tokens = torch.tensor([[self.tgt_pdx]]).to(self.device) # <sos>
            decoder_out = decode(prev_tgt_tokens, encoder_out, src_mask)

            best_k_probs, best_k_idx = decoder_out[:, -1, :].topk(beam_size)
            log_scores = torch.log(best_k_probs).view(beam_size) #-1也行

            gen_seqs = torch.cat((gen_seqs, best_k_idx.transpose(0, 1)), dim=1)

            encoder_out = encoder_out.repeat(beam_size, 1, 1)
            for step in range(2, self.max_seq_len):
                # 这里存在一个src_mask的广播
                decoder_out = decode(gen_seqs, encoder_out, src_mask)

                best_k2_probs, best_k2_idx = decoder_out[:, -1, :].topk(beam_size)

                log_scores_2 = torch.log(best_k2_probs).view(beam_size, -1) + log_scores.view(beam_size, 1)

                log_scores, best_k_idx_in_k2 = log_scores_2.view(-1).topk(beam_size)

                best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
                best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

                gen_seqs = torch.cat((gen_seqs, best_k_idx.unsqueeze(1)), dim=1)

                eos_locs = gen_seqs == self.tgt_eos_idx
                seq_lens, _ = len_map[:, :eos_locs.size(1)].masked_fill(~eos_locs, self.max_seq_len).min(1)

                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    #_, ans_idx = log_scores.max(0)
                    _, ans_idx = log_scores.div(seq_lens.type(torch.float) ** 0.005).max(0)
                    ans_idx = ans_idx.item()
                    break
        
        return ''.join([self.tgt_itos[s] for s in gen_seqs[ans_idx, :seq_lens[ans_idx]]])

    def translate(self, sentence: str, beam_size=8, src_lang='zh', tgt_lang='en'):
        jieba.setLogLevel(logging.INFO)
        if src_lang == 'zh':
            word_list = list(jieba.cut(sentence))
        else:
            word_list = sentence.split()
        with torch.no_grad():
            if beam_size == 1:
                return print(self._greedy_search(word_list), end="\n")
            else:
                return print(self._beam_search(word_list), end="\n")