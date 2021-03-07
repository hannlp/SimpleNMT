from torchtext import data, datasets
import dill
import torch


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class MyIterator(data.Iterator):
    # code from http://nlp.seas.harvard.edu/2018/04/03/attention.html
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class DataLoader(object):
    def __init__(self, PAD='<pad>', START='<sos>', END='<eos>', UNK='<unk>') -> None:
        self.PAD, self.START, self.END, self.UNK = PAD, START, END, UNK
        self.SRC = data.Field(pad_token=PAD)
        self.TGT = data.Field(init_token=START,
                              eos_token=END,
                              pad_token=PAD)

    def _save_dataloader(self, path):
        torch.save(self, path, pickle_module=dill)

    # TODO: 该函数中，随机划分会导致每次的词典不一样，从而导致之前训练好的模型不能加载，所以词典也要保存
    def load_translation(self, path, exts, batch_size=64, dl_save_path=None, device=None):
        DATA = datasets.TranslationDataset(
            path=path, exts=exts, fields=(('src', self.SRC), ('trg', self.TGT)))
        train, valid = DATA.split(split_ratio=0.95)

        self.SRC.build_vocab(train)
        self.TGT.build_vocab(train)
        self._save_dataloader(dl_save_path)

        train_iter = MyIterator(train, batch_size=batch_size, device=device,
                                repeat=False, sort_key=lambda x:
                                (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True,
                                shuffle=True)
        valid_iter = MyIterator(valid, batch_size=batch_size, device=device,
                                repeat=False, sort_key=lambda x:
                                (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True,
                                shuffle=True)

        return train_iter, valid_iter

    def load_tabular(path, format):
        pass


def prepare_batch(batch, CUDA_OK=False):
    src_tokens = batch.src.transpose(0, 1)
    prev_tgt_tokens = batch.trg.transpose(0, 1)[:, :-1]
    tgt_tokens = batch.trg.transpose(0, 1)[:, 1:]
    if CUDA_OK:
        src_tokens, prev_tgt_tokens, tgt_tokens = src_tokens.cuda(
        ), prev_tgt_tokens.cuda(), tgt_tokens.cuda()
    return src_tokens, prev_tgt_tokens, tgt_tokens
