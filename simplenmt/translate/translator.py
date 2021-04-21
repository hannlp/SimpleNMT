import time
import dill
import jieba
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import datasets
from models import build_model
from data.dataloader import MyIterator, batch_size_fn
from data.utils import prepare_batch
from .utils import de_numericalize
from .algorithms import beam_search, greedy_search


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
        self.beam_size = args.beam_size
        self.length_penalty = args.length_penalty
        self.max_seq_length = args.max_seq_length

        self.model = self._load_model(args.ckpt_path)
        self.model.eval()

    def _load_model(self, ckpt_path):
        '''
        checkpoint(dict):
            - epoch(int)
            - model(dict): model.state_dict()
            - settings(NameSpace): train_args
        '''

        checkpoint = torch.load(ckpt_path, map_location=self.device)

        model = build_model(checkpoint['settings'], use_cuda=torch.cuda.is_available())
        model.load_state_dict(checkpoint['model'])
        if hasattr(model, 'module'):
            model = model.module
        model.to(self.device)
        return model

    def generate(self, test_path, exts, batch_size=3200):
        test = datasets.TranslationDataset(
            path=test_path, exts=exts, 
            fields=(('src', self.dl.SRC), ('trg', self.dl.TGT)))
        
        test_iter = MyIterator(test, batch_size=batch_size, device=None,
                               repeat=False, sort_key=lambda x:
                               (len(x.src), len(x.trg)),
                               batch_size_fn=batch_size_fn, train=False,
                               shuffle=True)
        
        print('Writing result to {} ...'.format(test_path + '.result'))
        start_time = time.time()
        with open(test_path + '.result', 'w', encoding='utf8') as f, torch.no_grad():     
            for i, batch in enumerate(test_iter, start=1):
                #print("batch {}: preparing batch".format(i))
                src_tokens, _, tgt_tokens = prepare_batch(
                    batch, use_cuda=torch.cuda.is_available())

                src_sentences = de_numericalize(self.dl.SRC.vocab, src_tokens)
                tgt_sentences = de_numericalize(self.dl.TGT.vocab, tgt_tokens)
                
                if self.beam_size > 0:
                    pred_tokens, tgt_len = beam_search(model=self.model, 
                                            src_tokens=src_tokens,
                                            beam_size=self.beam_size,
                                            length_penalty=self.length_penalty,
                                            max_len=self.max_seq_length,
                                            bos=self.tgt_sos_idx,
                                            eos=self.tgt_eos_idx,
                                            pad=self.tgt_pdx)
                else:
                    pred_tokens = greedy_search(model=self.model,
                                                src_tokens=src_tokens,
                                                max_len=self.max_seq_length,
                                                bos=self.tgt_sos_idx,
                                                eos=self.tgt_eos_idx,
                                                pad=self.tgt_pdx)
                
                # TODO: 优化beam search 停止时间
                # print(len(tgt_len), tgt_len)
                # input()

                #pred_tokens = self.batch_greedy_search(src_tokens)
                pred_sentences = de_numericalize(self.dl.TGT.vocab, pred_tokens)

                for src_words, tgt_words, pred_words in zip(src_sentences, tgt_sentences, pred_sentences):
                    content = '-S\t{}\n-T\t{}\n-P\t{}\n\n'.format(
                        ' '.join(src_words), ' '.join(tgt_words), ' '.join(pred_words))            
                    f.write(content); print(content)

        print('Successful. Generate time:{:.1f} min, results were saved at{}'
                .format((time.time() - start_time) / 60, test_path + '.result'))

    def batch_greedy_search(self, src_tokens):
        batch_size = src_tokens.size(0)
        done = torch.tensor([False] * batch_size).to(self.device)
        
        encoder_out, src_mask = self._encode(src_tokens)

        gen_seqs = torch.full((batch_size, 1), self.tgt_sos_idx).to(self.device)
        # - gen_seqs: (batch_size, 1) -> <sos>

        probs = F.softmax(self._decode(gen_seqs, encoder_out, src_mask), dim=-1) # TODO: use log_softmax
        _, max_idxs = probs.topk(1) # new words
        
        for step in range(2, self.max_seq_length):           
            done = done | max_idxs.eq(self.tgt_eos_idx).squeeze() #TODO : stop rules
            if all(done):
                break
            
            gen_seqs = torch.cat((gen_seqs, max_idxs.to(self.device)), dim=1)
            # - gen_seqs: (batch_size, step) -> batch seqs

            probs = F.softmax(self._decode(gen_seqs, encoder_out, src_mask), dim=-1)
            _, max_idxs = probs.topk(1)
        
        return gen_seqs

    def _encode(self, src_tokens):
        src_mask = src_tokens.eq(self.src_pdx)
        encoder_out = self.model.encoder(src_tokens, src_mask)
        return encoder_out, src_mask

    def _decode(self, prev_tgt_tokens, encoder_out, src_mask):
        decoder_out = self.model.decoder(
            prev_tgt_tokens, encoder_out, src_mask, None)
        decoder_out = decoder_out[:,-1,:] # get last token
        model_out = self.model.out_vocab_proj(decoder_out)
        return model_out

    def translate(self, sentence: str, beam_size=8):
        jieba.setLogLevel(logging.INFO)
        # TODO: here need a tokenize function: STR -> word list
        word_list = [w for w in list(jieba.cut(sentence)) if w.strip()]

        with torch.no_grad():
            src_tokens = self.dl.SRC.numericalize([word_list], self.device) # (1, src_len)
            gen_seqs = self.batch_greedy_search(src_tokens)
            translated = de_numericalize(self.dl.TGT.vocab, gen_seqs)[0]
            print(' '.join(translated), end="\n")



'''
已弃用代码段
# temp(for transformer)
def _encode(self, src_tokens):
    src_mask = src_tokens.eq(self.src_pdx).to(self.device)
    encoder_out = self.model.encoder(src_tokens, src_mask)
    return encoder_out, src_mask

def _decode(self, prev_tgt_tokens, encoder_out, src_mask):
    tgt_mask = prev_tgt_tokens.eq(self.tgt_pdx).to(self.device)
    decoder_out = self.model.decoder(
        prev_tgt_tokens, encoder_out, src_mask, tgt_mask)
    decoder_out = decoder_out[:,-1,:] # get last token
    model_out = self.model.out_vocab_proj(decoder_out)
    return model_out

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
'''