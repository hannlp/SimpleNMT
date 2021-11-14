# SimpleNMT
A simple and readable **Neural Machine Translation** system

## 1 background
* The process of **automatic translation of natural language by a computer** is called **Machine Translation (MT)**.
* **Neural Machine Translation (NMT)** directly uses the Encoder-Decoder framework to perform end-to-end mapping of Distributed Representation language, which has the advantages of unified model structure and high translation quality, and has become the mainstream of the times.
* The development of machine translation is mainly attributed to the promotion of **open source systems** and **evaluation competitions**. There are many excellent neural machine translation systems ([fairseq](https://github.com/pytorch/fairseq), [OpenNMT](https://opennmt.net/), [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), etc.), but these open source systems have the disadvantages of complex implementation, too much redundant code, and difficult for beginners to read.

So, I am committed to building a Neural Machine Translation system that is easy to read, use, and friendly to beginners. (This is my graduation project ^ ^)

## 2 Quick Start
### 2.1 Download
```bash
git clone https://github.com/hannlp/SimpleNMT
cd SimpleNMT/simplenmt
pip install -r ../requirements.txt
```

### 2.2 Train your model
```bash
python train.py -data_path .. -save_path ..
```

use ```python train.py -h``` for more helps:
```bash
usage: train.py [-h] [-src SRC] [-tgt TGT] [-data_path DATA_PATH]
                [-save_path SAVE_PATH] [-batch_size BATCH_SIZE]
                [-max_seq_len MAX_SEQ_LEN] [-n_epochs N_EPOCHS]
                [-log_interval LOG_INTERVAL]
                [-keep_last_ckpts KEEP_LAST_CKPTS] [-optim OPTIM]
                [-model MODEL] [-d_model D_MODEL] [-n_layers N_LAYERS]
                [-share_vocab] [-p_drop P_DROP] [-lr LR] [-lr_scale LR_SCALE]
                [-betas BETAS [BETAS ...]] [-n_head N_HEAD]
                [-label_smoothing LABEL_SMOOTHING]
                [-warmup_steps WARMUP_STEPS] [-bidirectional]
                [-attn_type ATTN_TYPE] [-rnn_type RNN_TYPE]
```


### 2.3 Use your model to translate
```bash
python translate.py -data_path .. -save_path ..
```

use ```python translate.py -h``` for more helps:
```bash
usage: translate.py [-h] [-src SRC] [-tgt TGT] [-batch_size BATCH_SIZE]
                    [-data_path DATA_PATH] [-save_path SAVE_PATH]
                    [-ckpt_suffix CKPT_SUFFIX] [-max_seq_len MAX_SEQ_LEN]
                    [-generate] [-quiet] [-beam_size BEAM_SIZE]
                    [-length_penalty LENGTH_PENALTY]
```

## 3 Example
### 3.1 Train
This is a real example of using SimpleNMT to train a Chinese-English translation model. My parallel corpus is placed in ```/content/drive/MyDrive/Datasets/v15news/```, called ```train.zh```, ```train.en```,  ```valid.zh``` and ```valid.en```respectively. About the preprocessing method of parallel corpus, see this [blog](https://hannlp.github.io/2021-01-16-Use-fairseq-to-train-a-Chinese-English-translation-model-from-scratch/).
```
python train.py -src zh -tgt en -warmup_steps 8000 -data_path /content/drive/MyDrive/Datasets/v15news -save_path /content
```

This is the training process:  
```
21-05-02 08:39:58 | Loading train and valid data from '/content/drive/MyDrive/Datasets/v15news/train', '/content/drive/MyDrive/Datasets/v15news/valid', suffix:('.zh', '.en') ...
21-05-02 08:40:05 | Building src and tgt vocabs ...
21-05-02 08:40:08 | Vocab size | SRC(zh): 37,143 types, TGT(en): 30,767 types
21-05-02 08:40:09 | The dataloader has saved at '/content/zh-en.dl'
21-05-02 08:40:09 | Namespace(attn_type='general', batch_size=4096, betas=(0.9, 0.98), bidirectional=False, d_model=512, data_path='/content/drive/MyDrive/Datasets/v15news', keep_last_ckpts=5, label_smoothing=0.1, log_interval=100, lr=0.001, lr_scale=1.0, max_seq_len=2048, model='Transformer', n_epochs=40, n_head=8, n_layers=6, n_src_words=37143, n_tgt_words=30767, optim='noam', p_drop=0.1, rnn_type='gru', save_path='/content', share_vocab=False, src='zh', src_pdx=1, tgt='en', tgt_pdx=1, use_cuda=True, warmup_steps=8000)
21-05-02 08:40:17 | Params count | encoder: 37,932,544, decoder: 40,976,896, others: 15,783,471, total: 94,692,911
21-05-02 08:40:17 | Transformer(
  (encoder): Encoder(
    (dropout): Dropout(p=0.1, inplace=False)
    (input_embedding): Embedding(37143, 512, padding_idx=1)
    (positional_encode): PositionalEncode()
    (layers): ModuleList(
      (0): EncoderLayer(
        (dropout): Dropout(p=0.1, inplace=False)
        (sublayer1_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (self_attn): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=True)
          (w_k): Linear(in_features=512, out_features=512, bias=True)
          (w_v): Linear(in_features=512, out_features=512, bias=True)
          (w_out): Linear(in_features=512, out_features=512, bias=True)
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
        (self_attn): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=True)
          (w_k): Linear(in_features=512, out_features=512, bias=True)
          (w_v): Linear(in_features=512, out_features=512, bias=True)
          (w_out): Linear(in_features=512, out_features=512, bias=True)
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
    (input_embedding): Embedding(30767, 512, padding_idx=1)
    (positional_encode): PositionalEncode()
    (layers): ModuleList(
      (0): DecoderLayer(
        (dropout): Dropout(p=0.1, inplace=False)
        (sublayer1_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (masked_self_attn): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=True)
          (w_k): Linear(in_features=512, out_features=512, bias=True)
          (w_v): Linear(in_features=512, out_features=512, bias=True)
          (w_out): Linear(in_features=512, out_features=512, bias=True)
        )
        (sublayer2_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (context_attn): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=True)
          (w_k): Linear(in_features=512, out_features=512, bias=True)
          (w_v): Linear(in_features=512, out_features=512, bias=True)
          (w_out): Linear(in_features=512, out_features=512, bias=True)
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
        (masked_self_attn): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=True)
          (w_k): Linear(in_features=512, out_features=512, bias=True)
          (w_v): Linear(in_features=512, out_features=512, bias=True)
          (w_out): Linear(in_features=512, out_features=512, bias=True)
        )
        (sublayer2_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (context_attn): MultiHeadAttention(
          (w_q): Linear(in_features=512, out_features=512, bias=True)
          (w_k): Linear(in_features=512, out_features=512, bias=True)
          (w_v): Linear(in_features=512, out_features=512, bias=True)
          (w_out): Linear(in_features=512, out_features=512, bias=True)
        )
        (sublayer3_prenorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (pos_wise_ffn): FeedForwardNetwork(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
      )
    )
  )
  (out_vocab_proj): Linear(in_features=512, out_features=30767, bias=True)
)
21-05-02 08:40:55 | Epoch: 1, batch: [100/2140], lr: 0.000006, loss: 8.68375, ppl: 4574.50, acc: 5.90%, n_steps: 100
21-05-02 08:41:26 | Epoch: 1, batch: [200/2140], lr: 0.000012, loss: 7.62761, ppl: 1316.82, acc: 6.84%, n_steps: 200
21-05-02 08:41:57 | Epoch: 1, batch: [300/2140], lr: 0.000019, loss: 7.03944, ppl: 664.28, acc: 9.41%, n_steps: 300
21-05-02 08:42:28 | Epoch: 1, batch: [400/2140], lr: 0.000025, loss: 7.21272, ppl: 819.51, acc: 10.30%, n_steps: 400
21-05-02 08:42:59 | Epoch: 1, batch: [500/2140], lr: 0.000031, loss: 7.09707, ppl: 718.99, acc: 14.52%, n_steps: 500
21-05-02 08:43:31 | Epoch: 1, batch: [600/2140], lr: 0.000037, loss: 7.06151, ppl: 686.29, acc: 16.00%, n_steps: 600
21-05-02 08:44:02 | Epoch: 1, batch: [700/2140], lr: 0.000043, loss: 6.83975, ppl: 522.45, acc: 17.74%, n_steps: 700
21-05-02 08:44:33 | Epoch: 1, batch: [800/2140], lr: 0.000049, loss: 6.90300, ppl: 573.03, acc: 18.18%, n_steps: 800
21-05-02 08:45:05 | Epoch: 1, batch: [900/2140], lr: 0.000056, loss: 6.76517, ppl: 494.24, acc: 16.98%, n_steps: 900
21-05-02 08:45:36 | Epoch: 1, batch: [1000/2140], lr: 0.000062, loss: 6.63938, ppl: 427.50, acc: 19.78%, n_steps: 1000
21-05-02 08:46:07 | Epoch: 1, batch: [1100/2140], lr: 0.000068, loss: 5.75466, ppl: 154.18, acc: 28.80%, n_steps: 1100
21-05-02 08:46:38 | Epoch: 1, batch: [1200/2140], lr: 0.000074, loss: 6.07888, ppl: 218.73, acc: 23.01%, n_steps: 1200
21-05-02 08:47:09 | Epoch: 1, batch: [1300/2140], lr: 0.000080, loss: 5.83510, ppl: 169.73, acc: 26.62%, n_steps: 1300
21-05-02 08:47:40 | Epoch: 1, batch: [1400/2140], lr: 0.000086, loss: 6.06895, ppl: 222.28, acc: 23.06%, n_steps: 1400
21-05-02 08:48:12 | Epoch: 1, batch: [1500/2140], lr: 0.000093, loss: 5.63430, ppl: 133.63, acc: 27.65%, n_steps: 1500
21-05-02 08:48:43 | Epoch: 1, batch: [1600/2140], lr: 0.000099, loss: 5.57024, ppl: 121.16, acc: 28.52%, n_steps: 1600
21-05-02 08:49:14 | Epoch: 1, batch: [1700/2140], lr: 0.000105, loss: 6.09650, ppl: 224.54, acc: 23.51%, n_steps: 1700
21-05-02 08:49:45 | Epoch: 1, batch: [1800/2140], lr: 0.000111, loss: 5.45549, ppl: 108.25, acc: 30.28%, n_steps: 1800
21-05-02 08:50:17 | Epoch: 1, batch: [1900/2140], lr: 0.000117, loss: 5.84876, ppl: 170.74, acc: 25.03%, n_steps: 1900
21-05-02 08:50:48 | Epoch: 1, batch: [2000/2140], lr: 0.000124, loss: 5.56161, ppl: 124.00, acc: 29.38%, n_steps: 2000
21-05-02 08:51:19 | Epoch: 1, batch: [2100/2140], lr: 0.000130, loss: 5.67892, ppl: 140.96, acc: 26.16%, n_steps: 2100
21-05-02 08:51:40 | Valid | Epoch: 1, loss: 5.27728, ppl: 86.11, acc: 31.02%, elapsed: 11.4 min
21-05-02 08:52:25 | Epoch: 2, batch: [100/2140], lr: 0.000138, loss: 5.36792, ppl: 97.01, acc: 29.43%, n_steps: 2240
21-05-02 08:52:56 | Epoch: 2, batch: [200/2140], lr: 0.000145, loss: 5.36300, ppl: 97.59, acc: 29.87%, n_steps: 2340
...
21-05-02 09:02:17 | Epoch: 2, batch: [2000/2140], lr: 0.000256, loss: 4.31736, ppl: 29.59, acc: 42.65%, n_steps: 4140
21-05-02 09:02:48 | Epoch: 2, batch: [2100/2140], lr: 0.000262, loss: 4.94857, ppl: 58.46, acc: 34.57%, n_steps: 4240
21-05-02 09:03:08 | Valid | Epoch: 2, loss: 4.06680, ppl: 21.39, acc: 46.92%, elapsed: 11.4 min
21-05-02 09:03:52 | Epoch: 3, batch: [100/2140], lr: 0.000271, loss: 4.03694, ppl: 21.26, acc: 45.57%, n_steps: 4380
21-05-02 09:04:23 | Epoch: 3, batch: [200/2140], lr: 0.000277, loss: 3.72947, ppl: 15.09, acc: 51.72%, n_steps: 4480
...
21-05-02 09:13:44 | Epoch: 3, batch: [2000/2140], lr: 0.000388, loss: 3.59666, ppl: 12.80, acc: 53.22%, n_steps: 6280
21-05-02 09:14:16 | Epoch: 3, batch: [2100/2140], lr: 0.000394, loss: 3.45400, ppl: 10.89, acc: 55.09%, n_steps: 6380
21-05-02 09:14:36 | Valid | Epoch: 3, loss: 3.65300, ppl: 13.07, acc: 53.00%, elapsed: 11.4 min
21-05-02 09:15:21 | Epoch: 4, batch: [100/2140], lr: 0.000403, loss: 3.26954, ppl: 8.66, acc: 58.38%, n_steps: 6520
21-05-02 09:15:52 | Epoch: 4, batch: [200/2140], lr: 0.000409, loss: 3.51244, ppl: 11.62, acc: 53.65%, n_steps: 6620
...
21-05-02 09:25:14 | Epoch: 4, batch: [2000/2140], lr: 0.000482, loss: 3.74748, ppl: 15.40, acc: 52.17%, n_steps: 8420
21-05-02 09:25:45 | Epoch: 4, batch: [2100/2140], lr: 0.000479, loss: 3.82970, ppl: 16.94, acc: 49.77%, n_steps: 8520
21-05-02 09:26:06 | Valid | Epoch: 4, loss: 3.45292, ppl: 10.80, acc: 55.95%, elapsed: 11.4 min
21-05-02 09:26:50 | Epoch: 5, batch: [100/2140], lr: 0.000475, loss: 2.97675, ppl: 6.41, acc: 62.10%, n_steps: 8660
21-05-02 09:27:21 | Epoch: 5, batch: [200/2140], lr: 0.000472, loss: 2.93373, ppl: 6.08, acc: 64.58%, n_steps: 8760
...
21-05-02 09:36:45 | Epoch: 5, batch: [2000/2140], lr: 0.000430, loss: 3.11882, ppl: 7.47, acc: 59.97%, n_steps: 10560
21-05-02 09:37:16 | Epoch: 5, batch: [2100/2140], lr: 0.000428, loss: 3.13934, ppl: 7.59, acc: 59.95%, n_steps: 10660
21-05-02 09:37:36 | Valid | Epoch: 5, loss: 3.29597, ppl: 9.07, acc: 58.56%, elapsed: 11.4 min
21-05-02 09:38:20 | Epoch: 6, batch: [100/2140], lr: 0.000425, loss: 3.07922, ppl: 7.10, acc: 61.38%, n_steps: 10800
21-05-02 09:38:51 | Epoch: 6, batch: [200/2140], lr: 0.000423, loss: 2.91873, ppl: 5.87, acc: 63.37%, n_steps: 10900
...
21-05-02 09:48:13 | Epoch: 6, batch: [2000/2140], lr: 0.000392, loss: 2.95715, ppl: 6.30, acc: 63.82%, n_steps: 12700
21-05-02 09:48:45 | Epoch: 6, batch: [2100/2140], lr: 0.000391, loss: 3.03277, ppl: 6.73, acc: 61.75%, n_steps: 12800
21-05-02 09:49:05 | Valid | Epoch: 6, loss: 3.22096, ppl: 8.22, acc: 59.79%, elapsed: 11.4 min
21-05-02 09:49:50 | Epoch: 7, batch: [100/2140], lr: 0.000389, loss: 2.75536, ppl: 5.01, acc: 67.36%, n_steps: 12940
21-05-02 09:50:21 | Epoch: 7, batch: [200/2140], lr: 0.000387, loss: 3.21883, ppl: 8.28, acc: 60.00%, n_steps: 13040
...
21-05-02 09:59:43 | Epoch: 7, batch: [2000/2140], lr: 0.000363, loss: 2.89408, ppl: 5.83, acc: 65.01%, n_steps: 14840
21-05-02 10:00:14 | Epoch: 7, batch: [2100/2140], lr: 0.000362, loss: 3.30365, ppl: 9.40, acc: 56.28%, n_steps: 14940
21-05-02 10:00:35 | Valid | Epoch: 7, loss: 3.17850, ppl: 7.80, acc: 60.79%, elapsed: 11.4 min
21-05-02 10:01:19 | Epoch: 8, batch: [100/2140], lr: 0.000360, loss: 2.66141, ppl: 4.47, acc: 68.30%, n_steps: 15080
21-05-02 10:01:50 | Epoch: 8, batch: [200/2140], lr: 0.000359, loss: 2.65836, ppl: 4.44, acc: 68.47%, n_steps: 15180
...
21-05-02 10:11:12 | Epoch: 8, batch: [2000/2140], lr: 0.000339, loss: 3.06003, ppl: 7.08, acc: 61.69%, n_steps: 16980
21-05-02 10:11:43 | Epoch: 8, batch: [2100/2140], lr: 0.000338, loss: 2.93119, ppl: 6.05, acc: 64.74%, n_steps: 17080
21-05-02 10:12:04 | Valid | Epoch: 8, loss: 3.16045, ppl: 7.71, acc: 61.08%, elapsed: 11.4 min
21-05-02 10:12:48 | Epoch: 9, batch: [100/2140], lr: 0.000337, loss: 2.61907, ppl: 4.27, acc: 69.40%, n_steps: 17220
21-05-02 10:13:19 | Epoch: 9, batch: [200/2140], lr: 0.000336, loss: 2.56966, ppl: 3.97, acc: 70.58%, n_steps: 17320
21-05-02 10:13:50 | Epoch: 9, batch: [300/2140], lr: 0.000335, loss: 2.72842, ppl: 4.75, acc: 68.06%, n_steps: 17420
...
21-05-02 10:22:40 | Epoch: 9, batch: [2000/2140], lr: 0.000320, loss: 2.73787, ppl: 4.88, acc: 67.31%, n_steps: 19120
21-05-02 10:23:12 | Epoch: 9, batch: [2100/2140], lr: 0.000319, loss: 2.73082, ppl: 4.83, acc: 67.69%, n_steps: 19220
21-05-02 10:23:32 | Valid | Epoch: 9, loss: 3.17937, ppl: 7.71, acc: 61.20%, elapsed: 11.4 min
21-05-02 10:24:17 | Epoch: 10, batch: [100/2140], lr: 0.000318, loss: 2.64270, ppl: 4.31, acc: 68.17%, n_steps: 19360
21-05-02 10:24:49 | Epoch: 10, batch: [200/2140], lr: 0.000317, loss: 2.48681, ppl: 3.65, acc: 72.20%, n_steps: 19460
...
21-05-02 10:34:10 | Epoch: 10, batch: [2000/2140], lr: 0.000303, loss: 2.54778, ppl: 3.93, acc: 71.23%, n_steps: 21260
21-05-02 10:34:41 | Epoch: 10, batch: [2100/2140], lr: 0.000302, loss: 2.76253, ppl: 5.02, acc: 68.03%, n_steps: 21360
21-05-02 10:35:02 | Valid | Epoch: 10, loss: 3.16470, ppl: 7.71, acc: 61.64%, elapsed: 11.4 min
21-05-02 10:35:46 | Epoch: 11, batch: [100/2140], lr: 0.000301, loss: 2.40264, ppl: 3.30, acc: 74.37%, n_steps: 21500
21-05-02 10:36:17 | Epoch: 11, batch: [200/2140], lr: 0.000301, loss: 2.45678, ppl: 3.50, acc: 74.19%, n_steps: 21600
...
21-05-02 10:45:38 | Epoch: 11, batch: [2000/2140], lr: 0.000289, loss: 2.67876, ppl: 4.56, acc: 68.56%, n_steps: 23400
21-05-02 10:46:09 | Epoch: 11, batch: [2100/2140], lr: 0.000288, loss: 2.53057, ppl: 3.86, acc: 71.24%, n_steps: 23500
21-05-02 10:46:30 | Valid | Epoch: 11, loss: 3.17310, ppl: 7.79, acc: 61.70%, elapsed: 11.4 min
21-05-02 10:47:13 | Epoch: 12, batch: [100/2140], lr: 0.000287, loss: 2.31367, ppl: 2.99, acc: 75.61%, n_steps: 23640
21-05-02 10:47:44 | Epoch: 12, batch: [200/2140], lr: 0.000287, loss: 2.30427, ppl: 2.98, acc: 76.15%, n_steps: 23740
21-05-02 10:48:16 | Epoch: 12, batch: [300/2140], lr: 0.000286, loss: 2.34651, ppl: 3.07, acc: 76.04%, n_steps: 23840
21-05-02 10:48:47 | Epoch: 12, batch: [400/2140], lr: 0.000286, loss: 2.43701, ppl: 3.40, acc: 73.67%, n_steps: 23940
21-05-02 10:49:18 | Epoch: 12, batch: [500/2140], lr: 0.000285, loss: 2.47633, ppl: 3.55, acc: 71.92%, n_steps: 24040
21-05-02 10:49:49 | Epoch: 12, batch: [600/2140], lr: 0.000284, loss: 2.56293, ppl: 3.95, acc: 71.05%, n_steps: 24140
21-05-02 10:50:20 | Epoch: 12, batch: [700/2140], lr: 0.000284, loss: 2.44471, ppl: 3.43, acc: 73.47%, n_steps: 24240
21-05-02 10:50:51 | Epoch: 12, batch: [800/2140], lr: 0.000283, loss: 2.41029, ppl: 3.33, acc: 74.38%, n_steps: 24340
21-05-02 10:51:23 | Epoch: 12, batch: [900/2140], lr: 0.000283, loss: 2.45736, ppl: 3.52, acc: 72.67%, n_steps: 24440
21-05-02 10:51:54 | Epoch: 12, batch: [1000/2140], lr: 0.000282, loss: 2.51822, ppl: 3.78, acc: 72.50%, n_steps: 24540
21-05-02 10:52:25 | Epoch: 12, batch: [1100/2140], lr: 0.000282, loss: 2.47280, ppl: 3.54, acc: 72.88%, n_steps: 24640
21-05-02 10:52:56 | Epoch: 12, batch: [1200/2140], lr: 0.000281, loss: 2.47236, ppl: 3.57, acc: 72.93%, n_steps: 24740
21-05-02 10:53:28 | Epoch: 12, batch: [1300/2140], lr: 0.000280, loss: 2.38675, ppl: 3.20, acc: 74.37%, n_steps: 24840
21-05-02 10:53:59 | Epoch: 12, batch: [1400/2140], lr: 0.000280, loss: 2.44812, ppl: 3.54, acc: 72.66%, n_steps: 24940
21-05-02 10:54:30 | Epoch: 12, batch: [1500/2140], lr: 0.000279, loss: 2.50485, ppl: 3.73, acc: 71.41%, n_steps: 25040
21-05-02 10:55:01 | Epoch: 12, batch: [1600/2140], lr: 0.000279, loss: 3.02722, ppl: 6.81, acc: 63.36%, n_steps: 25140
21-05-02 10:55:32 | Epoch: 12, batch: [1700/2140], lr: 0.000278, loss: 2.58520, ppl: 4.11, acc: 69.30%, n_steps: 25240
21-05-02 10:56:03 | Epoch: 12, batch: [1800/2140], lr: 0.000278, loss: 3.25484, ppl: 8.77, acc: 60.83%, n_steps: 25340
21-05-02 10:56:34 | Epoch: 12, batch: [1900/2140], lr: 0.000277, loss: 2.47392, ppl: 3.58, acc: 71.74%, n_steps: 25440
21-05-02 10:57:06 | Epoch: 12, batch: [2000/2140], lr: 0.000277, loss: 2.66247, ppl: 4.43, acc: 69.41%, n_steps: 25540
21-05-02 10:57:37 | Epoch: 12, batch: [2100/2140], lr: 0.000276, loss: 2.63270, ppl: 4.28, acc: 69.21%, n_steps: 25640
21-05-02 10:57:58 | Valid | Epoch: 12, loss: 3.20003, ppl: 7.91, acc: 61.71%, elapsed: 11.4 min
```

### 3.2 Translate
After training the model, use the following command to use your best model for **interactive translation**:  
```bash
python translate.py -generate -src zh -tgt en -beam_size 5 -data_path /content/drive/MyDrive/Datasets/v15news -save_path /content
```

The translate results:
```bash
Please input a sentence(zh): 我 爱你。
i love you .

Please input a sentence(zh): 其中有一些政策被获得竞争优势的欲望所驱使，比如中国对绿色产业的支持。
some of these policies are driven by the desire to gain a competitive edge , such as china ’ s support for the green industry .

Please input a sentence(zh): 收入不平等可能再次开始扩大，尽管去年的中位家庭收入和贫困率指标有了重大改善。
income inequality is likely to start widening again , despite significant improvements in household income and the poverty rate last year .

Please input a sentence(zh): 在欧洲，知道目前银行才被明确要求解决资本短缺和杠杆问题，处理残留劣质资产。
in europe , it is known that banks are now required to address capital shortfalls and leverage problems and handle legacy assets .

Please input a sentence(zh): 须知民粹主义并不是某个心怀恶意的外来势力强加于欧洲身上的；而是在欧洲内部有机地滋生，并在真实存在且广泛的不满情绪的推动下蔓延开来。
populism is not imposed on europe by a malicious foreign power ; it is an air of organic tolerance within europe that is real and widespread .
```

It can also **generate translations** in batches for evaluation, which requires a test set in ```/content/drive/MyDrive/Datasets/v15news/```, called ```test.zh``` and ```test.en``` respectively
```bash
python translate.py -generate -src zh -tgt en -data_path /content/drive/MyDrive/Datasets/v15news -save_path /content
```

The generate preocess:  
```bash
# /content/result.txt
-S	此外 ， 化石 燃料 成本 随 油价 涨@@ 跌 而 剧烈 波动 ， 而且 核电 及 火电厂 的 集中 分布 也 为 电能 输送 设置 了 障碍 。
-T	moreover , fossil-fuel costs fluctuate wildly with oil prices , and the centralized nature of nuclear and coal-fired power stations creates distribution problems .
-P	moreover , the cost of fossil fuels has soared and plummeted , and the concentration of nuclear power and coal-fired power plants has created barriers to electricity delivery .

-S	肯尼迪 的 许多 顾问 和 美国 军方 领导人 敦促 他 采取 空袭 和 入侵 ， 现在 我们 知道 ， 这 有可能 导致 苏联 指挥官 动用 战术 核武器 。
-T	many of kennedy ’ s advisers , as well as us military leaders , urged an air strike and invasion , which we now know might have led soviet field commanders to use their tactical nuclear weapons .
-P	many of kennedy ’ s advisers and us military leaders urged him to adopt airstrikes and incursions , and now we know that it could lead to the use of tactical nuclear weapons by soviet commanders .

-S	事实上 ， 倘若 美国 真 能 跟 中国 谈判 以 减少 对 美 贸易顺差 ， 就 得 增加 与其 他 一些 国家 的 逆差 以 弥补 中间 的 差额 。
-T	and , in fact , if the us managed to negotiate a reduction in , say , china ’ s trade surplus vis-à-vis the us , the us would simply have to increase its deficit with some other country to make up for it .
-P	indeed , if the us were serious about negotiating with china to reduce its trade surplus with the us , it would have to increase its deficit with some other countries to make up for the difference .
```

### 3.3 Evaluation and comparison
And then, you can use the script to evaluate the translation results:
```bash
grep ^-T /content/result.txt | cut -f2 > ref.txt
grep ^-P /content/result.txt | cut -f2 > pred.txt
sed -r 's/(@@ )| (@@ ?$)//g' < pred.txt  > pred1.txt
sed -r 's/(@@ )| (@@ ?$)//g' < ref.txt  > ref1.txt
perl utils/multi-bleu.perl pred1.txt < ref1.txt
```

The evaluate result:
```
BLEU = 25.80, 58.7/31.7/19.7/12.6 (BP=0.991, ratio=0.991, hyp_len=194341, ref_len=196061)
```

The following is the evaluation result on the model trained using [fairseq](https://github.com/pytorch/fairseq), it can be found that the two are very close
```
BLEU = 26.06, 59.3/32.5/20.3/13.2 (BP=0.972, ratio=0.972, hyp_len=190618, ref_len=196075)
```

# 4 Acknowledgement
These are the projects I have referred to during the implementation process, thank them very much.
1. [bentrevett/pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq)
2. [harvardnlp/annotated-transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
3. [jadore801120/attention-is-all-you-need](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
4. [fairseq](https://github.com/pytorch/fairseq), [OpenNMT](https://opennmt.net/), [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), etc.