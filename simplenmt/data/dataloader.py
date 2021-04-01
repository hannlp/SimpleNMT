import dill
import torch
from torchtext.legacy import data, datasets
from .constants import Constants


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
    def __init__(self) -> None:
        self.PAD = Constants.PAD
        self.START = Constants.START
        self.END = Constants.END
        self.UNK = Constants.UNK
        self.SRC = data.Field(pad_token=Constants.PAD)
        self.TGT = data.Field(init_token=Constants.START,
                              eos_token=Constants.END,
                              pad_token=Constants.PAD)

    def load_translation(self, exts, data_path=None, train_path=None, valid_path=None, 
                         split_ratio=0.95, batch_size=64, dl_save_path=None,
                         share_vocab=False):

        if data_path:
            print("Loading parallel corpus from \'{}\', \'{}\' ...".format(data_path + exts[0], data_path + exts[1]), end=" ")
            DATA = datasets.TranslationDataset(
                path=data_path, exts=exts, fields=(('src', self.SRC), ('trg', self.TGT)))
            print("Successful.")

            train, valid = DATA.split(split_ratio=split_ratio)
        else:
            print("Loading train data and valid data from \'{}\', \'{}\' ...".format(train_path, valid_path), end=" ")
            train = datasets.TranslationDataset(
                path=train_path, exts=exts, fields=(('src', self.SRC), ('trg', self.TGT)))
            valid = datasets.TranslationDataset(
                path=valid_path, exts=exts, fields=(('src', self.SRC), ('trg', self.TGT)))
            print("Successful.")

        print("Building src and tgt vocabs ...", end=" ")
        if not share_vocab:
            self.SRC.build_vocab(train.src)
            self.TGT.build_vocab(train.trg)
        else:
            self.SRC.build_vocab(train.src, train.trg)
            self.TGT.vocab = self.SRC.vocab
        print("Successful.")
        self._add_index()

        torch.save(self, dl_save_path, pickle_module=dill)
        print("The dataloader is saved at \'{}\'".format(dl_save_path))

        train_iter = MyIterator(train, batch_size=batch_size, device=None,
                                repeat=False, sort_key=lambda x:
                                (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True,
                                shuffle=True)
        valid_iter = MyIterator(valid, batch_size=batch_size, device=None,
                                repeat=False, sort_key=lambda x:
                                (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True,
                                shuffle=True)

        return train_iter, valid_iter
        
    def _add_index(self):
        self.src_padding_index = self.SRC.vocab.stoi[Constants.PAD]
        self.tgt_padding_index = self.TGT.vocab.stoi[Constants.PAD]

    def load_tabular(self, path, format):
        pass
