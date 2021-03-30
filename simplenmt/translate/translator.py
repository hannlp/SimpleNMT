import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import dill
import jieba
import logging
from torchtext import datasets
from translate.beam import Beam
from models import build_model
from data.dataloader import MyIterator, batch_size_fn
from data.utils import prepare_batch
from data.constants import Constants

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
            map_location=self.device)

        model = build_model(checkpoint['settings'], use_cuda=torch.cuda.is_available())
        model.load_state_dict(checkpoint['model'])
        if hasattr(model, 'module'):
            model = model.module
        model.to(self.device)
        return model

    def generate(self, test_path, exts, batch_size=3200):

        def de_numericalize(vocab, tokens, remove_constants=
                            {Constants.PAD, Constants.START, Constants.END}):
            
            sentences = []
            for ex in tokens:
                end, words_list = False, []
                for x in ex:
                    word = vocab.itos[x]
                    end = True if word == Constants.END else end
                    if word not in remove_constants and not end:
                        words_list.append(word)
                    else:
                        pass
                sentences.append(words_list)

            return sentences

        test = datasets.TranslationDataset(
            path=test_path, exts=exts, 
            fields=(('src', self.dl.SRC), ('trg', self.dl.TGT)))
        
        test_iter = MyIterator(test, batch_size=batch_size, device=None,
                                repeat=False, sort_key=lambda x:
                                (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=False,
                                shuffle=True)
       
        with open(test_path + '.result', 'w', encoding='utf8') as f:
            start_time = time.time()
            print('Writing result to {} ...'.format(test_path + '.result'))
            with torch.no_grad():
                for i, batch in enumerate(test_iter, start=1):
                    #print("batch {}: preparing batch".format(i))
                    src_tokens, _, tgt_tokens = prepare_batch(
                        batch, use_cuda=torch.cuda.is_available())

                    src_sentences = de_numericalize(self.dl.SRC.vocab, src_tokens)
                    tgt_sentences = de_numericalize(self.dl.TGT.vocab, tgt_tokens)
                    
                    #print("start batch greedy search")
                    #pred_tokens = self.batch_beam_search(src_tokens, beam_size=4)
                    pred_tokens = self.batch_greedy_search(src_tokens)
                    #print("end batch greedy search")

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
            #print(step, done)
            if all(done):
                break
            
            gen_seqs = torch.cat((gen_seqs, max_idxs.to(self.device)), dim=1)
            # - gen_seqs: (batch_size, step) -> batch seqs

            probs = F.softmax(self._decode(gen_seqs, encoder_out, src_mask), dim=-1)
            _, max_idxs = probs.topk(1)
        
        return gen_seqs

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

    #TODO :to fix this fucntion
    def batch_beam_search(self, src_tokens, beam_size=4):
        ''' Translation work in one batch '''
        
        # Batch size is in different location depending on data.
        batch_size = src_tokens.size(0) # src_tokens: [batch_size x src_len]

        # Encode
        encoder_out, src_mask =  self._encode(src_tokens)

        # Repeat data for beam
        src_tokens = src_tokens.repeat(1, beam_size).view(batch_size * beam_size, -1)
        src_mask = src_mask.repeat(1, beam_size).view(batch_size * beam_size, -1)

        encoder_out =  encoder_out.repeat(1, beam_size, 1).view(
            batch_size * beam_size, encoder_out.size(1), encoder_out.size(2))

        # Prepare beams
        beams = [Beam(beam_size, pdx=self.tgt_pdx, bos_idx=self.tgt_sos_idx, eos_idx=self.tgt_eos_idx) for _ in range(batch_size)]
        beam_inst_idx_map = {
            beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))
        }
        n_remaining_sents = batch_size

        print("开始解码了")
        # Decode
        for i in range(self.max_seq_length):
            len_dec_seq = i + 1
            # Preparing decoded data_seq
            # size: [batch_size x beam_size x seq_len]
            dec_partial_inputs = torch.stack([
                b.get_current_state() for b in beams if not b.done])
            # size: [batch_size * beam_size x seq_len]
            dec_partial_inputs = dec_partial_inputs.view(-1, len_dec_seq)

            print("解到第{}步了".format(len_dec_seq))
            # Decoding
            model_out = self._decode(dec_partial_inputs, encoder_out, src_mask)
            out = F.log_softmax(model_out, dim=-1)

            # [batch_size x beam_size x tgt_vocab_size]
            word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

            active_beam_idx_list = []
            for beam_idx in range(batch_size):
                if beams[beam_idx].done:
                    continue

                inst_idx = beam_inst_idx_map[beam_idx] # 해당 beam_idx 의 데이터가 실제 data 에서 몇번째 idx인지
                if not beams[beam_idx].advance(word_lk.data[inst_idx]):
                    active_beam_idx_list += [beam_idx]

            if not active_beam_idx_list: # all instances have finished their path to <eos>
                break

            # In this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_inst_idxs = torch.LongTensor(
                [beam_inst_idx_map[k] for k in active_beam_idx_list]) # TODO: fix

            # update the idx mapping
            beam_inst_idx_map = {
                beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}

            def update_active_seq(seq_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''
                inst_idx_dim_size, *rest_dim_sizes = seq_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_seq_data = seq_var.data.view(n_remaining_sents, -1)
                active_seq_data = original_seq_data.index_select(0, active_inst_idxs)
                active_seq_data = active_seq_data.view(*new_size)
                return active_seq_data

            def update_active_enc_info(enc_info_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''

                inst_idx_dim_size, *rest_dim_sizes = enc_info_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_enc_info_data = enc_info_var.data.view(
                    n_remaining_sents, -1, self.model.d_model)
                active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs)
                active_enc_info_data = active_enc_info_data.view(*new_size)
                return active_enc_info_data

            src_tokens = update_active_seq(src_tokens, active_inst_idxs)
            encoder_out = update_active_enc_info(encoder_out, active_inst_idxs)

            # update the remaining size
            n_remaining_sents = len(active_inst_idxs)

        # Return useful information
        all_hyp, all_scores = [], []
        n_best = 8

        for beam_idx in range(batch_size):
            scores, tail_idxs = beams[beam_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [beams[beam_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]

        return all_hyp, all_scores

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

    def translate(self, sentence: str, beam_size=8):
        jieba.setLogLevel(logging.INFO)
        word_list = [w for w in list(jieba.cut(sentence)) if w.strip()]
        with torch.no_grad():
            if beam_size == 1:
                return print(self._greedy_search(word_list), end="\n")
            else:
                return print(self._beam_search(word_list), end="\n")