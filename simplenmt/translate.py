import argparse
import dill

def main():
    dl = torch.load(dl_save_path, pickle_module=dill)
    translator = Translator(args, Transformer, EN_ZH,
                            load_path='/home/hanyuchen/NMT/checkpoints')
    translator.generate(valid_iter)

if __name__ == '__main__':
    main()