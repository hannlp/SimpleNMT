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

| | d | d_ff | h |l_enc|l_dec| best(beam 1) | best(beam 5)| average last 5(beam 5)
|:----:| :-----:|:----: | :----: | :----: |:----: |:----: | :----: | :----:|
|small| 512 | 1024 | 4 |  6 | 6 | 31.77/1.0min|33.10/5.9min|**34.16**/5.3min
|base| 512|2048| 8| 6 | 6 |31.37/1.0min|32.67/6.0min|33.97/5.9min
|small (deep enc shallow dec)|512|1024|4|12|1|30.99/0.4min|32.28/3.7min|33.16/3.1min|
|base (deep enc shallow dec)|512|2048|8|12|1|30.22/0.4min|31.65/3.5min|32.88/2.8min|
|small (deep-norm)|512|1024|4|6|6|31.91|-|33.92|
|middle (deep-norm)|512|1024|4|24|6|32.01|-|34.07|
|deep (deep-norm)|256|1024|4|48|6|31.88|-|33.69|
|deep+ (deep-norm)|128|512|4|128|6|30.88|-|32.95|
|deep++ (deep-norm)|64|256|4|**200**|6|28.88|-|30.73|

