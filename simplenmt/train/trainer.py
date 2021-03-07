import torch
import time


class Trainer(object):
    def __init__(self, model, optimizer, criterion, warmup_steps=4000, lr_scal=1, d_model=512) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.warmup_steps = warmup_steps
        self.lr_scal = lr_scal
        self.d_model = d_model
        self._num_step = 0

    def train(self, train_iter, valid_iter, n_epochs, save_path=None):
        # TODO: 在训练前打印各种有用信息

        self._num_step = 0
        best_valid_loss = 1e9

        # - trianing ...
        for epoch in range(1, n_epochs + 1):
            is_best_epoch = False
            start_time = time.time()
            self._train_epoch(train_iter, epoch)

            if valid_iter != None:
                valid_loss = self._valid_epoch(valid_iter)
                self._print_log(epoch, valid_loss, start_time)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    is_best_epoch = True

            self._save_model(epoch, save_path, is_best_epoch)

    def _print_log(self, epoch, valid_loss, start_time):
        print("Valid | Epoch:{}, loss:{:.5}, training_time:{:.1f} min".format(
            epoch, valid_loss, (time.time() - start_time) / 60))

    def _train_epoch(self, train_iter, epoch):
        self.model.train()
        n_batches = len(list(iter(train_iter)))
        for i, batch in enumerate(train_iter, start=1):
            self._lr_step_update()
            self.optimizer.zero_grad()
            src_tokens, prev_tgt_tokens, tgt_tokens = prepare_batch(
                batch, CUDA_OK)
            out = self.model(src_tokens, prev_tgt_tokens)
            loss = self.criterion(
                out.reshape(-1, out.size(-1)), tgt_tokens.contiguous().view(-1))
            loss.backward()

            self.optimizer.step()
            if i % 100 == 0:
                print('{} | Epoch: {}, batch: [{}/{}], lr:{:.5}, loss: {:.5}'
                      .format(time.strftime("%y-%m-%d %H:%M:%S", time.localtime()),
                              epoch, i, n_batches, self._get_lr(), loss.item()))

    def _valid_epoch(self, valid_iter):
        self.model.eval()
        n_batches = len(list(iter(valid_iter)))
        with torch.no_grad():
            loss_list = []
            for _, batch in enumerate(valid_iter, start=1):
                src_tokens, prev_tgt_tokens, tgt_tokens = prepare_batch(
                    batch, CUDA_OK)
                out = self.model(src_tokens, prev_tgt_tokens)
                loss = self.criterion(
                    out.reshape(-1, out.size(-1)), tgt_tokens.contiguous().view(-1))
                loss_list.append(loss)
        return sum(loss_list) / n_batches

    def _save_model(self, epoch, path, is_best_epoch):
        checkpoint = {'epoch': epoch, 'model': self.model.state_dict()}
        torch.save(checkpoint, '{}/checkpoint_{}.pt'.format(path, epoch))
        if is_best_epoch:
            best_checkpoint = {'epoch': epoch,
                               'model': self.model.state_dict()}
            torch.save(best_checkpoint, '{}/checkpoint_best.pt'.format(path))

    def _lr_step_update(self):
        self._num_step += 1
        lrate = self.d_model ** -0.5 * \
            min(self._num_step ** -0.5, self._num_step * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_scal * lrate

    def _get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
