import os
import dill
import torch
from torchtext.legacy import data, datasets
from .constants import Constants


global max_src_in_batch, max_tgt_in_batch

# code from http://nlp.seas.harvard.edu/2018/04/03/attention.html
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

# code from http://nlp.seas.harvard.edu/2018/04/03/attention.html
class MyIterator(data.Iterator):
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
        self.SRC = data.Field(pad_token=Constants.PAD, 
                              batch_first=True)
        self.TGT = data.Field(init_token=Constants.START,
                              eos_token=Constants.END,
                              pad_token=Constants.PAD, 
                              batch_first=True)

    def load_translation(self, src, tgt, data_path=None, split_ratio=0.95, 
                        batch_size=64, dl_save_path=None, share_vocab=False):

        exts = ('.' + src, '.' + tgt) # default: ('.zh', '.en')
        if os.path.isdir(data_path):
            train_path, valid_path = data_path + '/train', data_path + '/valid'
            print("Loading train and valid data from \'{}\', \'{}\', suffix:{}' ...".format(
                train_path, valid_path, exts), end=" ")
            train = datasets.TranslationDataset(
                path=train_path, exts=exts, fields=(('src', self.SRC), ('trg', self.TGT)))
            valid = datasets.TranslationDataset(
                path=valid_path, exts=exts, fields=(('src', self.SRC), ('trg', self.TGT)))
            print("Successful.")
        else:
            print("Loading parallel corpus from \'{}\', suffix:{} ...".format(data_path, exts), end=" ")
            DATA = datasets.TranslationDataset(
                path=data_path, exts=exts, fields=(('src', self.SRC), ('trg', self.TGT)))
            train, valid = DATA.split(split_ratio=split_ratio)
            print("Successful.")

        print("Building src and tgt vocabs ...", end=" ")
        if not share_vocab:
            self.SRC.build_vocab(train.src)
            self.TGT.build_vocab(train.trg)
        else:
            self.SRC.build_vocab(train.src, train.trg)
            self.TGT.vocab = self.SRC.vocab
        print("Successful. ", end=" ")

        self.src_padding_index = self.SRC.vocab.stoi[Constants.PAD]
        self.tgt_padding_index = self.TGT.vocab.stoi[Constants.PAD]

        dl_path = '{}/{}-{}.dl'.format(dl_save_path, src, tgt)
        torch.save(self, dl_path, pickle_module=dill)
        print("The dataloader has saved at \'{}\'".format(dl_path))

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
        

    def load_tabular(self, path, format):
        pass
