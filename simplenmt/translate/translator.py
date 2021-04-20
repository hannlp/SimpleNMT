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
from .algorithms import BeamHypotheses


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
                
                # pred_tokens, _ = generate_beam(model=self.model, 
                #                             src_tokens=src_tokens,
                #                             beam_size=4,
                #                             length_penalty=1.0,
                #                             max_len=self.max_seq_length,
                #                             bos=self.tgt_sos_idx,
                #                             eos=self.tgt_eos_idx,
                #                             pad=self.tgt_pdx)
                pred_tokens = self.batch_beam_search(src_tokens=src_tokens,
                                                     beam_size=self.beam_size,
                                                     length_penalty=1.0)

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

    def batch_beam_search(self, src_tokens, beam_size, length_penalty):
        # batch size
        bs = len(src_tokens)

        src_enc, src_mask = self._encode(src_tokens)

        # expand to beam size the source latent representations
        src_enc = src_enc.repeat_interleave(beam_size, dim=0)
        src_mask = src_mask.repeat_interleave(beam_size, dim=0)

        # generated sentences (batch with beam current hypotheses)
        generated = src_tokens.new(bs * beam_size, self.max_seq_length).fill_(self.tgt_pdx)  # upcoming output
        generated[:, 0].fill_(self.tgt_sos_idx)

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, self.max_seq_length, length_penalty) for _ in range(bs)]

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # done sentences
        done = [False] * bs

        while cur_len < self.max_seq_length:

            # compute word scores
            model_out = self._decode(generated[:, :cur_len], src_enc, src_mask) # log softmax
            # - model_out: (batch_size * beam_size, vocab_size)
            scores = F.log_softmax(model_out, dim=-1)       # (bs * beam_size, n_words)
            n_words = scores.size(-1)
            # - scores: (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores.view(bs * beam_size, 1)      # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)             # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=-1, largest=True, sorted=True)
            # - next_scores, next_words: (bs, 2 * beam_size)

            # next batch beam content
            next_batch_beam = []  # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.tgt_pdx, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.tgt_eos_idx or cur_len + 1 == self.max_seq_length:
                        generated_hyps[sent_id].add(generated[sent_id * beam_size + beam_id, :cur_len].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.tgt_pdx, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)

            # sanity check / prepare next batch
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_tokens.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[beam_idx, :]
            generated[:, cur_len] = beam_words

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # select the best hypotheses
        tgt_len = src_tokens.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_tokens.new(bs, tgt_len.max().item()).fill_(self.tgt_pdx)
        for i, hypo in enumerate(best):
            decoded[i, :tgt_len[i] - 1] = hypo
            decoded[i, tgt_len[i] - 1] = self.tgt_eos_idx

        return decoded

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