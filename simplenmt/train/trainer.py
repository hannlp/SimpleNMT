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
        # TODO: 在训练前打印各种有用信息
        self.logger.info(self.model)
        self._num_steps = 0
        best_valid_loss = 1e9

        # - trianing ...
        for epoch in range(1, n_epochs + 1):
            is_best_epoch = False
            start_time = time.time()
            self._train_epoch(train_iter, epoch, log_interval)

            valid_loss = self._valid_epoch(valid_iter)
            self.logger.info("Valid | Epoch: {}, loss: {:.5}, ppl: {:.5}, elapsed: {:.1f} min, num_steps: {}".format(
                epoch, valid_loss, math.exp(valid_loss), (time.time() - start_time) / 60, self._num_steps))
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                is_best_epoch = True

            if ckpt_save_path is not None:
                self._save_model(epoch, ckpt_save_path, is_best_epoch)

    def _train_epoch(self, train_iter, epoch, log_interval):
        self.model.train()
        n_batches = len(list(iter(train_iter)))
        for i, batch in enumerate(train_iter, start=1):
            self._lr_step_update()
            self.optimizer.zero_grad()
            src_tokens, prev_tgt_tokens, tgt_tokens = prepare_batch(
                batch, use_cuda=self.use_cuda)
            out = self.model(src_tokens, prev_tgt_tokens)
            loss = self.criterion(
                out.reshape(-1, out.size(-1)), tgt_tokens.contiguous().view(-1))
            loss.backward()

            self.optimizer.step()
            if i % log_interval == 0:
                self.logger.info('Epoch: {}, batch: [{}/{}], lr: {:.5}, loss: {:.5}, ppl: {:.5}'
                      .format(epoch, i, n_batches, self._get_lr(), loss.item(), math.exp(loss.item())))

    def _valid_epoch(self, valid_iter):
        self.model.eval()
        n_batches = len(list(iter(valid_iter)))
        with torch.no_grad():
            loss_list = []
            for _, batch in enumerate(valid_iter, start=1):
                src_tokens, prev_tgt_tokens, tgt_tokens = prepare_batch(
                    batch, use_cuda=self.use_cuda)
                out = self.model(src_tokens, prev_tgt_tokens)
                loss = self.criterion(
                    out.reshape(-1, out.size(-1)), tgt_tokens.contiguous().view(-1))
                loss_list.append(loss.item())
        return sum(loss_list) / n_batches

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

    def _lr_step_update(self):
        self._num_steps += 1
        lrate = self.d_model ** -0.5 * \
            min(self._num_steps ** -0.5, self._num_steps * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_scale * lrate

    def _get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
