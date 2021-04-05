# SimpleNMT
A simple and readable **Neural Machine Translation** system

## 1 background
* The process of **automatic translation of natural language by a computer** is called **Machine Translation (MT)**.
* **Neural Machine Translation (NMT)** directly uses the Encoder-Decoder framework to perform end-to-end mapping of Distributed Representation language, which has the advantages of unified model structure and high translation quality, and has become the mainstream of the times.
* The development of machine translation is mainly attributed to the promotion of **open source systems** and **evaluation competitions**. There are many excellent neural machine translation systems ([fairseq](https://github.com/pytorch/fairseq), [OpenNMT](https://opennmt.net/), [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), etc.), but these open source systems have the disadvantages of complex implementation, too much redundant code, and difficult for beginners to read.

## 2 To do
I am committed to building a Neural Machine Translation system that is easy to read, use, and friendly to beginners.

## 3 Quick Start
### 3.1 Download
```bash
git clone https://github.com/hannlp/SimpleNMT
cd SimpleNMT/simplenmt
pip install -r ../requirements.txt
```

### 3.2 Train your model
```bash
python train.py -data_path .. -dl_path .. -ckpt_path ..
```

### 3.3 Use your model to translate
```bash
python translate.py -dl_path .. -ckpt_path ..
```

## 4 Example
### 4.1 Train
This is a real example of using SimpleNMT to train a Chinese-English translation model. My parallel corpus is placed in ```/content/drive/MyDrive/Datasets/v15news/```, called ```train.zh```, ```train.en```,  ```valid.zh``` and ```valid.en```respectively. About the preprocessing method of parallel corpus, see this [blog](https://hannlp.github.io/2021-01-16-Use-fairseq-to-train-a-Chinese-English-translation-model-from-scratch/).
```
python train.py -src zh -tgt en -train_path /content/drive/MyDrive/Datasets/v15news/train -valid_path /content/drive/MyDrive/Datasets/v15news/valid -dl_path /content/drive/MyDrive/v15news/zh_en.dl -ckpt_path /content/drive/MyDrive/v15news -batch_size 9600 -n_epochs 15
```

This is the training process:  
```
Loading train data and valid data from '/content/drive/MyDrive/Datasets/v15news/train', '/content/drive/MyDrive/Datasets/v15news/valid' ... Successful.
Building src and tgt vocabs ... Successful.
The dataloader is saved at '/content/drive/MyDrive/v15news/zh_en.dl'
Namespace(batch_size=9600, betas=(0.9, 0.98), ckpt_path='/content/drive/MyDrive/v15news', d_model=512, data_path='', dl_path='/content/drive/MyDrive/v15news/zh_en.dl', label_smoothing=0.1, lr=0.001, max_seq_len=2048, model='Transformer', n_epochs=15, n_head=8, n_layers=6, n_src_words=37127, n_tgt_words=30891, p_drop=0.1, share_vocab=False, src='zh', src_pdx=1, tgt='en', tgt_pdx=1, train_path='/content/drive/MyDrive/Datasets/v15news/train', valid_path='/content/drive/MyDrive/Datasets/v15news/valid', warmup_steps=4000)
Transformer(
  (encoder): Encoder(
    (dropout): Dropout(p=0.1, inplace=False)
    (input_embedding): Embedding(37127, 512, padding_idx=1)
    (positional_encode): PositionalEncode()
    (layers): ModuleList(
      (0): EncoderLayer(
        (dropout): Dropout(p=0.1, inplace=False)
        (sublayer1_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (multi_head_attention): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=False)
          (w_k): Linear(in_features=512, out_features=512, bias=False)
          (w_v): Linear(in_features=512, out_features=512, bias=False)
          (w_out): Linear(in_features=512, out_features=512, bias=False)
        )
        (sublayer2_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (pos_wise_ffn): FeedForwardNetwork(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
      )
      ...
      (5): EncoderLayer(
        (dropout): Dropout(p=0.1, inplace=False)
        (sublayer1_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (multi_head_attention): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=False)
          (w_k): Linear(in_features=512, out_features=512, bias=False)
          (w_v): Linear(in_features=512, out_features=512, bias=False)
          (w_out): Linear(in_features=512, out_features=512, bias=False)
        )
        (sublayer2_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (pos_wise_ffn): FeedForwardNetwork(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
      )
    )
    (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  )
  (decoder): Decoder(
    (dropout): Dropout(p=0.1, inplace=False)
    (input_embedding): Embedding(30891, 512, padding_idx=1)
    (positional_encode): PositionalEncode()
    (layers): ModuleList(
      (0): DecoderLayer(
        (dropout): Dropout(p=0.1, inplace=False)
        (sublayer1_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (masked_multi_head_attention): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=False)
          (w_k): Linear(in_features=512, out_features=512, bias=False)
          (w_v): Linear(in_features=512, out_features=512, bias=False)
          (w_out): Linear(in_features=512, out_features=512, bias=False)
        )
        (sublayer2_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (multi_head_attention): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=False)
          (w_k): Linear(in_features=512, out_features=512, bias=False)
          (w_v): Linear(in_features=512, out_features=512, bias=False)
          (w_out): Linear(in_features=512, out_features=512, bias=False)
        )
        (sublayer3_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (pos_wise_ffn): FeedForwardNetwork(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
      )
      ...
      (5): DecoderLayer(
        (dropout): Dropout(p=0.1, inplace=False)
        (sublayer1_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (masked_multi_head_attention): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=False)
          (w_k): Linear(in_features=512, out_features=512, bias=False)
          (w_v): Linear(in_features=512, out_features=512, bias=False)
          (w_out): Linear(in_features=512, out_features=512, bias=False)
        )
        (sublayer2_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (multi_head_attention): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=False)
          (w_k): Linear(in_features=512, out_features=512, bias=False)
          (w_v): Linear(in_features=512, out_features=512, bias=False)
          (w_out): Linear(in_features=512, out_features=512, bias=False)
        )
        (sublayer3_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (pos_wise_ffn): FeedForwardNetwork(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
      )
    )
  )
  (out_vocab_proj): Linear(in_features=512, out_features=30891, bias=True)
)
21-04-03 05:54:47 | Epoch: 1, batch: [100/938], lr: 1.7469e-05, loss: 7.5993, ppl: 1996.7
21-04-03 05:55:23 | Epoch: 1, batch: [200/938], lr: 3.4939e-05, loss: 7.3019, ppl: 1483.0
21-04-03 05:55:59 | Epoch: 1, batch: [300/938], lr: 5.2408e-05, loss: 7.5435, ppl: 1888.4
21-04-03 05:56:35 | Epoch: 1, batch: [400/938], lr: 6.9877e-05, loss: 6.9109, ppl: 1003.1
21-04-03 05:57:12 | Epoch: 1, batch: [500/938], lr: 8.7346e-05, loss: 6.6026, ppl: 737.04
21-04-03 05:57:48 | Epoch: 1, batch: [600/938], lr: 0.00010482, loss: 6.3106, ppl: 550.37
21-04-03 05:58:24 | Epoch: 1, batch: [700/938], lr: 0.00012228, loss: 6.4993, ppl: 664.66
21-04-03 05:59:00 | Epoch: 1, batch: [800/938], lr: 0.00013975, loss: 6.2946, ppl: 541.62
21-04-03 05:59:36 | Epoch: 1, batch: [900/938], lr: 0.00015722, loss: 6.3037, ppl: 546.58
Valid | Epoch: 1, loss: 5.9043, ppl: 366.61, elapsed: 5.9 min
21-04-03 06:00:43 | Epoch: 2, batch: [100/938], lr: 0.00018133, loss: 5.444, ppl: 231.37
21-04-03 06:01:19 | Epoch: 2, batch: [200/938], lr: 0.0001988, loss: 5.6483, ppl: 283.81
21-04-03 06:01:55 | Epoch: 2, batch: [300/938], lr: 0.00021627, loss: 4.7674, ppl: 117.62
21-04-03 06:02:33 | Epoch: 2, batch: [400/938], lr: 0.00023374, loss: 5.1199, ppl: 167.31
21-04-03 06:03:09 | Epoch: 2, batch: [500/938], lr: 0.00025121, loss: 4.9192, ppl: 136.89
21-04-03 06:03:45 | Epoch: 2, batch: [600/938], lr: 0.00026868, loss: 4.7982, ppl: 121.29
21-04-03 06:04:21 | Epoch: 2, batch: [700/938], lr: 0.00028615, loss: 4.9823, ppl: 145.82
21-04-03 06:04:58 | Epoch: 2, batch: [800/938], lr: 0.00030362, loss: 4.7086, ppl: 110.9
21-04-03 06:05:34 | Epoch: 2, batch: [900/938], lr: 0.00032109, loss: 4.8606, ppl: 129.11
Valid | Epoch: 2, loss: 4.5595, ppl: 95.534, elapsed: 5.9 min
21-04-03 06:06:40 | Epoch: 3, batch: [100/938], lr: 0.00034519, loss: 4.4961, ppl: 89.664
21-04-03 06:07:15 | Epoch: 3, batch: [200/938], lr: 0.00036266, loss: 4.4539, ppl: 85.962
21-04-03 06:07:52 | Epoch: 3, batch: [300/938], lr: 0.00038013, loss: 4.9643, ppl: 143.21
21-04-03 06:08:27 | Epoch: 3, batch: [400/938], lr: 0.0003976, loss: 4.1596, ppl: 64.044
21-04-03 06:09:04 | Epoch: 3, batch: [500/938], lr: 0.00041507, loss: 4.2274, ppl: 68.541
21-04-03 06:09:39 | Epoch: 3, batch: [600/938], lr: 0.00043254, loss: 4.1477, ppl: 63.286
21-04-03 06:10:15 | Epoch: 3, batch: [700/938], lr: 0.00045001, loss: 3.5308, ppl: 34.15
21-04-03 06:10:51 | Epoch: 3, batch: [800/938], lr: 0.00046748, loss: 4.029, ppl: 56.205
21-04-03 06:11:27 | Epoch: 3, batch: [900/938], lr: 0.00048495, loss: 3.8569, ppl: 47.318
Valid | Epoch: 3, loss: 3.8937, ppl: 49.093, elapsed: 5.8 min
21-04-03 06:12:33 | Epoch: 4, batch: [100/938], lr: 0.00050905, loss: 3.2992, ppl: 27.09
21-04-03 06:13:09 | Epoch: 4, batch: [200/938], lr: 0.00052652, loss: 3.8065, ppl: 44.994
21-04-03 06:13:45 | Epoch: 4, batch: [300/938], lr: 0.00054399, loss: 3.3751, ppl: 29.226
21-04-03 06:14:20 | Epoch: 4, batch: [400/938], lr: 0.00056146, loss: 3.5185, ppl: 33.735
21-04-03 06:14:57 | Epoch: 4, batch: [500/938], lr: 0.00057893, loss: 4.5561, ppl: 95.215
21-04-03 06:15:33 | Epoch: 4, batch: [600/938], lr: 0.0005964, loss: 3.6905, ppl: 40.063
21-04-03 06:16:08 | Epoch: 4, batch: [700/938], lr: 0.00061387, loss: 3.4279, ppl: 30.811
21-04-03 06:16:45 | Epoch: 4, batch: [800/938], lr: 0.00063134, loss: 3.6108, ppl: 36.995
21-04-03 06:17:20 | Epoch: 4, batch: [900/938], lr: 0.00064881, loss: 4.4326, ppl: 84.15
Valid | Epoch: 4, loss: 3.6058, ppl: 36.812, elapsed: 5.8 min
21-04-03 06:18:26 | Epoch: 5, batch: [100/938], lr: 0.00067292, loss: 3.0921, ppl: 22.024
21-04-03 06:19:02 | Epoch: 5, batch: [200/938], lr: 0.00069039, loss: 3.8602, ppl: 47.476
21-04-03 06:19:38 | Epoch: 5, batch: [300/938], lr: 0.00069427, loss: 3.2486, ppl: 25.754
21-04-03 06:20:14 | Epoch: 5, batch: [400/938], lr: 0.00068586, loss: 3.5188, ppl: 33.745
21-04-03 06:20:50 | Epoch: 5, batch: [500/938], lr: 0.00067775, loss: 3.2284, ppl: 25.24
21-04-03 06:21:26 | Epoch: 5, batch: [600/938], lr: 0.00066992, loss: 3.3343, ppl: 28.058
21-04-03 06:22:02 | Epoch: 5, batch: [700/938], lr: 0.00066235, loss: 3.4624, ppl: 31.892
21-04-03 06:22:37 | Epoch: 5, batch: [800/938], lr: 0.00065503, loss: 4.1181, ppl: 61.441
21-04-03 06:23:14 | Epoch: 5, batch: [900/938], lr: 0.00064796, loss: 3.6578, ppl: 38.775
Valid | Epoch: 5, loss: 3.4348, ppl: 31.026, elapsed: 5.8 min
21-04-03 06:24:21 | Epoch: 6, batch: [100/938], lr: 0.00063855, loss: 3.1628, ppl: 23.638
21-04-03 06:24:57 | Epoch: 6, batch: [200/938], lr: 0.00063199, loss: 3.6315, ppl: 37.768
21-04-03 06:25:34 | Epoch: 6, batch: [300/938], lr: 0.00062563, loss: 3.1879, ppl: 24.238
21-04-03 06:26:10 | Epoch: 6, batch: [400/938], lr: 0.00061945, loss: 3.0669, ppl: 21.476
21-04-03 06:26:46 | Epoch: 6, batch: [500/938], lr: 0.00061345, loss: 3.0862, ppl: 21.893
21-04-03 06:27:21 | Epoch: 6, batch: [600/938], lr: 0.00060763, loss: 3.1527, ppl: 23.4
21-04-03 06:27:57 | Epoch: 6, batch: [700/938], lr: 0.00060196, loss: 3.2172, ppl: 24.958
21-04-03 06:28:34 | Epoch: 6, batch: [800/938], lr: 0.00059646, loss: 2.9256, ppl: 18.645
21-04-03 06:29:09 | Epoch: 6, batch: [900/938], lr: 0.0005911, loss: 3.027, ppl: 20.635
Valid | Epoch: 6, loss: 3.3159, ppl: 27.547, elapsed: 5.9 min
21-04-03 06:30:15 | Epoch: 7, batch: [100/938], lr: 0.00058393, loss: 2.7524, ppl: 15.68
21-04-03 06:30:51 | Epoch: 7, batch: [200/938], lr: 0.0005789, loss: 3.0642, ppl: 21.417
21-04-03 06:31:27 | Epoch: 7, batch: [300/938], lr: 0.000574, loss: 2.8816, ppl: 17.843
21-04-03 06:32:03 | Epoch: 7, batch: [400/938], lr: 0.00056922, loss: 2.8788, ppl: 17.793
21-04-03 06:32:39 | Epoch: 7, batch: [500/938], lr: 0.00056455, loss: 3.0443, ppl: 20.995
21-04-03 06:33:15 | Epoch: 7, batch: [600/938], lr: 0.00056, loss: 3.2723, ppl: 26.371
21-04-03 06:33:50 | Epoch: 7, batch: [700/938], lr: 0.00055556, loss: 3.2764, ppl: 26.482
21-04-03 06:34:27 | Epoch: 7, batch: [800/938], lr: 0.00055122, loss: 3.2826, ppl: 26.646
21-04-03 06:35:03 | Epoch: 7, batch: [900/938], lr: 0.00054698, loss: 2.9471, ppl: 19.051
Valid | Epoch: 7, loss: 3.2457, ppl: 25.68, elapsed: 5.9 min
21-04-03 06:36:09 | Epoch: 8, batch: [100/938], lr: 0.00054129, loss: 2.6793, ppl: 14.574
21-04-03 06:36:45 | Epoch: 8, batch: [200/938], lr: 0.00053728, loss: 2.832, ppl: 16.979
21-04-03 06:37:21 | Epoch: 8, batch: [300/938], lr: 0.00053335, loss: 2.9928, ppl: 19.941
21-04-03 06:37:57 | Epoch: 8, batch: [400/938], lr: 0.00052951, loss: 2.9814, ppl: 19.715
21-04-03 06:38:34 | Epoch: 8, batch: [500/938], lr: 0.00052575, loss: 2.8535, ppl: 17.348
21-04-03 06:39:10 | Epoch: 8, batch: [600/938], lr: 0.00052207, loss: 3.1035, ppl: 22.275
21-04-03 06:39:46 | Epoch: 8, batch: [700/938], lr: 0.00051846, loss: 2.9771, ppl: 19.63
21-04-03 06:40:21 | Epoch: 8, batch: [800/938], lr: 0.00051493, loss: 2.9742, ppl: 19.573
21-04-03 06:40:57 | Epoch: 8, batch: [900/938], lr: 0.00051147, loss: 2.9323, ppl: 18.771
Valid | Epoch: 8, loss: 3.2259, ppl: 25.176, elapsed: 5.9 min
21-04-03 06:42:04 | Epoch: 9, batch: [100/938], lr: 0.00050681, loss: 2.8907, ppl: 18.006
21-04-03 06:42:40 | Epoch: 9, batch: [200/938], lr: 0.00050351, loss: 2.7374, ppl: 15.447
21-04-03 06:43:15 | Epoch: 9, batch: [300/938], lr: 0.00050027, loss: 2.7775, ppl: 16.078
21-04-03 06:43:51 | Epoch: 9, batch: [400/938], lr: 0.0004971, loss: 2.7274, ppl: 15.293
21-04-03 06:44:27 | Epoch: 9, batch: [500/938], lr: 0.00049398, loss: 2.9197, ppl: 18.535
21-04-03 06:45:03 | Epoch: 9, batch: [600/938], lr: 0.00049093, loss: 2.6709, ppl: 14.453
21-04-03 06:45:40 | Epoch: 9, batch: [700/938], lr: 0.00048792, loss: 2.8753, ppl: 17.73
21-04-03 06:46:15 | Epoch: 9, batch: [800/938], lr: 0.00048498, loss: 2.6451, ppl: 14.085
21-04-03 06:46:52 | Epoch: 9, batch: [900/938], lr: 0.00048208, loss: 2.5538, ppl: 12.856
Valid | Epoch: 9, loss: 3.2068, ppl: 24.699, elapsed: 5.9 min
21-04-03 06:48:00 | Epoch: 10, batch: [100/938], lr: 0.00047817, loss: 2.5319, ppl: 12.577
21-04-03 06:48:36 | Epoch: 10, batch: [200/938], lr: 0.0004754, loss: 2.5545, ppl: 12.865
21-04-03 06:49:12 | Epoch: 10, batch: [300/938], lr: 0.00047267, loss: 2.5001, ppl: 12.183
21-04-03 06:49:48 | Epoch: 10, batch: [400/938], lr: 0.00046999, loss: 2.5965, ppl: 13.417
21-04-03 06:50:24 | Epoch: 10, batch: [500/938], lr: 0.00046736, loss: 2.5852, ppl: 13.266
21-04-03 06:51:00 | Epoch: 10, batch: [600/938], lr: 0.00046476, loss: 2.6526, ppl: 14.19
21-04-03 06:51:36 | Epoch: 10, batch: [700/938], lr: 0.00046222, loss: 2.8505, ppl: 17.296
21-04-03 06:52:12 | Epoch: 10, batch: [800/938], lr: 0.00045971, loss: 2.721, ppl: 15.195
21-04-03 06:52:48 | Epoch: 10, batch: [900/938], lr: 0.00045724, loss: 2.7036, ppl: 14.934
Valid | Epoch: 10, loss: 3.2293, ppl: 25.262, elapsed: 5.9 min
21-04-03 06:53:52 | Epoch: 11, batch: [100/938], lr: 0.0004539, loss: 2.5226, ppl: 12.461
21-04-03 06:54:28 | Epoch: 11, batch: [200/938], lr: 0.00045153, loss: 2.4941, ppl: 12.11
21-04-03 06:55:04 | Epoch: 11, batch: [300/938], lr: 0.00044919, loss: 2.5072, ppl: 12.271
21-04-03 06:55:40 | Epoch: 11, batch: [400/938], lr: 0.00044688, loss: 2.5332, ppl: 12.593
21-04-03 06:56:16 | Epoch: 11, batch: [500/938], lr: 0.00044462, loss: 2.3391, ppl: 10.372
21-04-03 06:56:52 | Epoch: 11, batch: [600/938], lr: 0.00044238, loss: 2.6163, ppl: 13.685
21-04-03 06:57:28 | Epoch: 11, batch: [700/938], lr: 0.00044018, loss: 2.6116, ppl: 13.621
21-04-03 06:58:03 | Epoch: 11, batch: [800/938], lr: 0.00043802, loss: 2.5893, ppl: 13.321
21-04-03 06:58:39 | Epoch: 11, batch: [900/938], lr: 0.00043588, loss: 2.7065, ppl: 14.976
Valid | Epoch: 11, loss: 3.2399, ppl: 25.53, elapsed: 5.8 min
```

## 4.2 Translate
After training the model, use the following command to use your best model for **interactive translation**:  
```bash
python translate.py -dl_path /content/drive/MyDrive/v15news/zh_en.dl -ckpt_path /content/drive/MyDrive/v15news/checkpoint_best.pt
```

The translate results:
```bash
Please input a sentence of zh:其中有一些政策被获得竞争优势的欲望所驱使，比如中国对绿色产业的支持。
some of these policies are driven by the desire to gain a competitive advantage , as China supports the green sector .

Please input a sentence of zh:在欧洲，知道目前银行才被明确要求解决资本短缺和杠杆问题，处理残留劣质资产。
in Europe , knowing that the current bank is explicitly to address its capital shortfalls and leverage problems , and to treat residual assets with residual residual status .

Please input a sentence of zh:收入不平等可能再次开始扩大，尽管去年的中位家庭收入和贫困率指标有了重大改善。
income inequality may start widening again , though the indices for median household income and poverty improved dramatically during the past year .

Please input a sentence of zh:须知民粹主义并不是某个心怀恶意的外来势力强加于欧洲身上的；而是在欧洲内部有机地滋生，并在真实存在且广泛的不满情绪的推动下蔓延开来。
populism is not a malicious external actor in Europe ; it is a source of organic power in Europe , and spreading from real and widespread disaffection .
```

It can also **generate translations** in batches for evaluation, which requires a test set in ```/content/drive/MyDrive/Datasets/v15news/```  
```bash
python translate.py -src zh -tgt en -generate -test_path /content/drive/MyDrive/Datasets/v15news/test -dl_path /content/drive/MyDrive/v15news/zh_en.dl -ckpt_path /content/drive/MyDrive/v15news/checkpoint_best.pt
```

The generate preocess:  
```bash
# test.result
-S	当 利比亚 的 卡扎菲 威胁 要 把 反对派 批评者 像 &quot; 老鼠 &quot; 一样 杀掉 时 ， 联合国 军 根据 新 的 全球 信条 - - 保护 责任 （ respon@@ sib@@ ility to prot@@ ect ） 实施 了 干预 。
-T	when Libya &apos;s Muammar el-Qaddafi threatened to kill his rebellious detractors like &quot; rats , &quot; a UN coalition intervened under an emerging global doctrine : the responsibility to protect .
-P	when Libya &apos;s Muammar el-Qaddafi threatened to kill his opponents , like &quot; rats , &quot; the UN force intervened in accordance with a new global creed , which protects the responsibility ( R2P ) .


-S	比如 美国 最大 的 网上 汽车 租赁 公司 Z@@ ip@@ car 可以 让 人们 分享 使用 同 一@@ 辆车 （ 但愿 在 每个 中国 人 都 拥有 自己 的 私@@ 家@@ 车 之前 ， 他们 能 发现 这个 行业 ！
-T	Zi@@ p@@ car , for example , lets people share cars . ( let &apos;s hope the Chinese discover this before everyone in China buys their own car ! )
-P	for example , the largest US auto rental company Z@@ ic@@ o@@ i@@ o@@ st could enable people to share the same vehicle ( which would , it would be possible , before everyone Chinese owned their own private cars ! ) could find the industry .


-S	生于 埃及 、 现@@ 居 悉尼 的 谢赫 · 希拉里 （ She@@ ik H@@ il@@ al@@ y ） 有 可能 只是 描述 了 自己 潜意识 中 的 差别 意识 ， 并 错误 地 把 这种 意识 当成 了 洞察力 。
-T	Egyp@@ ti@@ an-@@ born She@@ ik Hil@@ aly , in Sydney , may have been ver@@ b@@ alizing a latent sense of other@@ ness and mistaking it for insight .
-P	She@@ ik She@@ ik , a charismatic and present , is probably merely a description of the difference in his unconscious nature , and a false sense of this awareness .


-S	在 联合国 的 相关 机构 宣称 人 道 努力 在 2000 年 避免 了 埃塞俄比亚 大 范围 的 饥荒 发生 之后 ， &quot; 人 道 政策 集团 &quot; 在 2004 年 的 一份 报告 中 引述 了 一项 调查 。
-T	a 2004 report by the Humanitarian Policy Group cited a survey carried out in Ethiopia after UN agencies said that humanitarian efforts had averted widespread famine in 2000 .
-P	after the UN &apos;s related agency , the human policy group declared that the humanitarian effort to avert Ethiopia &apos;s broad famine in 2000 , the &quot; humanitarian policy group &quot; cited a survey in a 2004 report .
```