# SimpleNMT
A simple and readable **Neural Machine Translation** system

## 1 background
* The process of **automatic translation of natural language by a computer** is called **Machine Translation (MT)**.
* **Neural Machine Translation (NMT)** directly uses the Encoder-Decoder framework to perform end-to-end mapping of Distributed Representation language, which has the advantages of unified model structure and high translation quality, and has become the mainstream of the times.
* The development of machine translation is mainly attributed to the promotion of **open source systems** and **evaluation competitions**. There are many excellent neural machine translation systems ([fairseq](https://github.com/pytorch/fairseq), [OpenNMT](https://opennmt.net/), [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), etc.), but these open source systems have the disadvantages of complex implementation, too much redundant code, and difficult for beginners to read.

## 2 To do
I am committed to building a Neural Machine Translation system that is easy to read, use, and friendly to beginners.

## 3 Quick to use
### 3.1 Download
```bash
git clone https://github.com/hannlp/SimpleNMT
cd SimpleNMT/simplenmt
```

### 3.2 Train your model
```bash
python train.py -data_path .. -dl_path .. -ckpt_path ..
```

## 4 Example
This is a real example of using SimpleNMT to train a Chinese-English translation model. My parallel corpus is placed in ```/content/drive/MyDrive/```, called ```clean.zh``` and ```clean.en``` respectively.
```
python train.py -data_path /content/drive/MyDrive/clean -dl_path /content/drive/MyDrive/zh_en.dl -ckpt_path /content/drive/MyDrive -batch_size 3200 -n_epochs 5
```

This is the training process:  
```
Namespace(batch_size=3200, betas=(0.9, 0.98), ckpt_path='/content/drive/MyDrive', cuda_ok=True, d_model=512, data_path='/content/drive/MyDrive/clean', dl_path='/content/drive/MyDrive/zh_en.dl', lr=0.001, model='Transformer', n_epochs=5, n_head=8, n_layer=6, n_src_words=37112, n_tgt_words=30891, p_drop=0.1, src='zh', src_pdx=1, tgt='en', tgt_pdx=1, warmup_steps=4000)
Transformer(
  (encoder): Encoder(
    (dropout): Dropout(p=0.1, inplace=False)
    (input_embedding): Embedding(37112, 512, padding_idx=1)
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
      (1): EncoderLayer(
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
      (2): EncoderLayer(
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
      (3): EncoderLayer(
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
      (4): EncoderLayer(
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
      (1): DecoderLayer(
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
      (2): DecoderLayer(
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
      (3): DecoderLayer(
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
      (4): DecoderLayer(
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
    (vocab_proj): Linear(in_features=512, out_features=30891, bias=True)
  )
)
21-03-09 07:00:08 | Epoch: 1, batch: [100/2717], lr:1.7469e-05, loss: 7.3134
21-03-09 07:01:25 | Epoch: 1, batch: [200/2717], lr:3.4939e-05, loss: 6.993
21-03-09 07:02:42 | Epoch: 1, batch: [300/2717], lr:5.2408e-05, loss: 6.5239
21-03-09 07:03:59 | Epoch: 1, batch: [400/2717], lr:6.9877e-05, loss: 7.0627
21-03-09 07:05:16 | Epoch: 1, batch: [500/2717], lr:8.7346e-05, loss: 6.1477
21-03-09 07:06:32 | Epoch: 1, batch: [600/2717], lr:0.00010482, loss: 6.0476
21-03-09 07:07:49 | Epoch: 1, batch: [700/2717], lr:0.00012228, loss: 6.0057
21-03-09 07:09:06 | Epoch: 1, batch: [800/2717], lr:0.00013975, loss: 5.6957
21-03-09 07:10:23 | Epoch: 1, batch: [900/2717], lr:0.00015722, loss: 5.8262
21-03-09 07:11:40 | Epoch: 1, batch: [1000/2717], lr:0.00017469, loss: 5.1887
21-03-09 07:12:57 | Epoch: 1, batch: [1100/2717], lr:0.00019216, loss: 5.1317
21-03-09 07:14:14 | Epoch: 1, batch: [1200/2717], lr:0.00020963, loss: 5.0596
21-03-09 07:15:31 | Epoch: 1, batch: [1300/2717], lr:0.0002271, loss: 5.1706
21-03-09 07:16:49 | Epoch: 1, batch: [1400/2717], lr:0.00024457, loss: 4.8406
21-03-09 07:18:06 | Epoch: 1, batch: [1500/2717], lr:0.00026204, loss: 5.7437
21-03-09 07:19:23 | Epoch: 1, batch: [1600/2717], lr:0.00027951, loss: 4.5875
21-03-09 07:20:40 | Epoch: 1, batch: [1700/2717], lr:0.00029698, loss: 4.5823
21-03-09 07:21:57 | Epoch: 1, batch: [1800/2717], lr:0.00031445, loss: 4.928
21-03-09 07:23:15 | Epoch: 1, batch: [1900/2717], lr:0.00033192, loss: 4.3869
21-03-09 07:24:32 | Epoch: 1, batch: [2000/2717], lr:0.00034939, loss: 4.2969
21-03-09 07:25:49 | Epoch: 1, batch: [2100/2717], lr:0.00036685, loss: 4.1936
21-03-09 07:27:06 | Epoch: 1, batch: [2200/2717], lr:0.00038432, loss: 4.036
21-03-09 07:28:24 | Epoch: 1, batch: [2300/2717], lr:0.00040179, loss: 3.533
21-03-09 07:29:41 | Epoch: 1, batch: [2400/2717], lr:0.00041926, loss: 4.1269
21-03-09 07:30:58 | Epoch: 1, batch: [2500/2717], lr:0.00043673, loss: 3.2661
21-03-09 07:32:14 | Epoch: 1, batch: [2600/2717], lr:0.0004542, loss: 3.3929
21-03-09 07:33:30 | Epoch: 1, batch: [2700/2717], lr:0.00047167, loss: 4.1443
Valid | Epoch:1, loss:3.6985, training_time:35.7 min
21-03-09 07:36:05 | Epoch: 2, batch: [100/2717], lr:0.00049211, loss: 2.8699
```