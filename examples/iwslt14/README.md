# Iwslt14.de-en

Change the ```data_dir``` to your preprocessed iwslt14.de-en datasets, and change the ```root_dir``` for yourself

My dir tree:
```bash
~
└── mt
    ├── SimpleNMT
    ├── data
        └── iwslt14.de-en
    └── checkpoints
```

train:
```bash
bash train.sh [--device DEVICE --exp_tag EXP_TAG ...]
```

evaluate:
```bash
bash evaluate.sh [--device DEVICE --beam_size BEAM_SIZE --n_average N_AVERAGE ...]
```

result(multi-bleu.perl):

| | d_model | d_ff | n_head |l_enc|l_dec| best(beam 1) | best(beam 5)| average last 5(beam 5)
|:----:| :-----:|:----: | :----: | :----: |:----: |:----: | :----: | :----:|
|small| 512 | 1024 | 4 |  6 | 6 | 31.77/1.0min|33.10/5.9min|34.16/5.3min
|base| 512|2048| 8| 6 | 6 |31.37/1.0min|32.67/6.0min|33.97/5.9min
|small (deep enc shallow dec)|512|1024|4|12|1|30.99/0.4min|32.28/3.7min|33.16/3.1min|
|base (deep enc shallow dec)|512|2048|8|12|1|30.22/0.4min|31.65/3.5min|32.88/2.8min|
|base (big ffn deep enc shallow dec)|512|4096|8|12|1|30.05/0.6min|31.48/3.9min|32.82/3.8min|