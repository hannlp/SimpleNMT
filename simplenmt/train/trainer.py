import os
import torch
import time
import math
from data.utils import prepare_batch

class Trainer(object):
    def __init__(self, args, model, optimizer, criterion, lr_scale=1, logger=None) -> None:
        self.use_cuda = args.use_cuda
        self.settings = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.warmup_steps = args.warmup_steps
        self.lr_scale = lr_scale
        self.d_model = args.d_model
        self._num_steps = 0
        self.logger = logger
        self.ckpt_queue = list()
        self.queue_size = args.keep_last_ckpts

    def train(self, train_iter, valid_iter, n_epochs, log_interval=100, ckpt_save_path=None):
        """ Begin trianing ..."""

        # TODO: 这里可以实现一下加载last模型实现继续训练
        self.logger.info(self.model)
        self._num_steps = 0
        best_valid_loss = 1e9

        for epoch in range(1, n_epochs + 1):
            is_best_epoch = False
            start_time = time.time()
            self._train_epoch(train_iter, epoch, log_interval)

            loss_per_word, accuracy = self._valid_epoch(valid_iter)
            self.logger.info("Valid | Epoch: {}, loss: {:.5}, ppl: {:.5}, acc: %{:.2}, elapsed: {:.1f} min, num_steps: {}".format(
                epoch, loss_per_word, math.exp(loss_per_word), accuracy, (time.time() - start_time) / 60, self._num_steps))
            
            if loss_per_word < best_valid_loss:
                best_valid_loss = loss_per_word
                is_best_epoch = True

            if ckpt_save_path is not None:
                self._save_model(epoch, ckpt_save_path, is_best_epoch)

    def _train_epoch(self, train_iter, epoch, log_interval):
        self.model.train()
        n_batches = len(list(iter(train_iter)))

        for i, batch in enumerate(train_iter, start=1):
            self._num_steps += 1
            self.optimizer.zero_grad()
            src_tokens, prev_tgt_tokens, tgt_tokens = prepare_batch(
                batch, use_cuda=self.use_cuda)
            model_out = self.model(src_tokens, prev_tgt_tokens)
            loss, n_correct, n_word = self._cal_performance(pred=model_out, gold=tgt_tokens)
            loss.backward()
            self.optimizer.step()

            loss_per_word = loss.item() / n_word
            acc = n_correct / n_word
            if i % log_interval == 0:
                self.logger.info('Epoch: {}, batch: [{}/{}], lr: {:.5}, loss: {:.5}, ppl: {:.5}, acc: %{:.2}'
                        .format(epoch, i, n_batches, self._get_lr(), loss_per_word, math.exp(loss_per_word), acc * 100))

    def _valid_epoch(self, valid_iter):
        self.model.eval()
        total_loss, total_words, correct_words = 0, 0, 0

        with torch.no_grad():
            for batch in valid_iter:
                src_tokens, prev_tgt_tokens, tgt_tokens = prepare_batch(
                    batch, use_cuda=self.use_cuda)
                model_out = self.model(src_tokens, prev_tgt_tokens)
                loss, n_correct, n_word = self._cal_performance(pred=model_out, gold=tgt_tokens)
                
                total_loss += loss.item()
                total_words += n_word
                correct_words += n_correct

        loss_per_word = total_loss / total_words
        accuracy = correct_words / total_words
        return loss_per_word, accuracy

    def _cal_performance(self, pred, gold):
        # - pred: (batch_size, tgt_len, d_model), - gold: (batch_size, tgt_len)

        pred = pred.reshape(-1, pred.size(-1)) # (batch_size * tgt_len, d_model)
        gold = gold.contiguous().view(-1) # (batch_size * tgt_len)
        loss = self.criterion(pred, gold)
        
        tgt_pdx = self.criterion.ignore_index
        non_pad_mask = gold.ne(tgt_pdx)
        n_correct = pred.max(dim=-1).indices.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()

        return loss, n_correct, n_word

    def _save_model(self, epoch, ckpt_save_path, is_best_epoch):
        '''
        checkpoint(dict):
            - epoch(int)
            - model(dict): model.state_dict()
            - settings(NameSpace): train_args
        '''
        checkpoint = {'epoch': epoch, 'model': self.model.state_dict(), 'settings': self.settings}
        torch.save(checkpoint, '{}/checkpoint_{}.pt'.format(ckpt_save_path, epoch))
        self.ckpt_queue.append(epoch)
        if len(self.ckpt_queue) > self.queue_size:
            ckpt_suffix = self.ckpt_queue.pop(0)
            to_del_ckpt = '{}/checkpoint_{}.pt'.format(ckpt_save_path, ckpt_suffix)
            if os.path.exists(to_del_ckpt):
                os.remove(to_del_ckpt)

        # save the last checkpoint
        torch.save(checkpoint, '{}/checkpoint_last.pt'.format(ckpt_save_path))
        # save the best checkpoint
        if is_best_epoch:           
            torch.save(checkpoint, '{}/checkpoint_best.pt'.format(ckpt_save_path))

    def _get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
