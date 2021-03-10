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
This is a real example of using SimpleNMT to train a Chinese-English translation model. My parallel corpus is placed in ```/content/drive/MyDrive/```, called ```clean.zh``` and ```clean.en``` respectively.
```
python train.py -data_path /content/drive/MyDrive/clean -dl_path /content/drive/MyDrive/zh_en.dl -ckpt_path /content/drive/MyDrive -batch_size 6400 -n_epochs 5
```

This is the training process:  
```
Namespace(batch_size=6400, betas=(0.9, 0.98), ckpt_path='/content/drive/MyDrive', cuda_ok=True, d_model=512, data_path='/content/drive/MyDrive/clean', dl_path='/content/drive/MyDrive/zh_en.dl', lr=0.001, model='Transformer', n_epochs=10, n_head=8, n_layer=6, n_src_words=37125, n_tgt_words=30903, p_drop=0.1, src='zh', src_pdx=1, tgt='en', tgt_pdx=1, warmup_steps=4000)
Transformer(
  (encoder): Encoder(
    (dropout): Dropout(p=0.1, inplace=False)
    (input_embedding): Embedding(37125, 512, padding_idx=1)
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
    (input_embedding): Embedding(30903, 512, padding_idx=1)
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
    (vocab_proj): Linear(in_features=512, out_features=30903, bias=True)
  )
)
21-03-09 09:47:57 | Epoch: 1, batch: [100/1382], lr:1.7469e-05, loss: 6.948
21-03-09 09:48:38 | Epoch: 1, batch: [200/1382], lr:3.4939e-05, loss: 6.8318
21-03-09 09:49:19 | Epoch: 1, batch: [300/1382], lr:5.2408e-05, loss: 7.0079
21-03-09 09:50:01 | Epoch: 1, batch: [400/1382], lr:6.9877e-05, loss: 6.3954
21-03-09 09:50:42 | Epoch: 1, batch: [500/1382], lr:8.7346e-05, loss: 6.1232
21-03-09 09:51:23 | Epoch: 1, batch: [600/1382], lr:0.00010482, loss: 5.7747
21-03-09 09:52:04 | Epoch: 1, batch: [700/1382], lr:0.00012228, loss: 5.6869
21-03-09 09:52:45 | Epoch: 1, batch: [800/1382], lr:0.00013975, loss: 5.3506
21-03-09 09:53:26 | Epoch: 1, batch: [900/1382], lr:0.00015722, loss: 5.2774
21-03-09 09:54:07 | Epoch: 1, batch: [1000/1382], lr:0.00017469, loss: 5.1601
21-03-09 09:54:48 | Epoch: 1, batch: [1100/1382], lr:0.00019216, loss: 6.1068
21-03-09 09:55:30 | Epoch: 1, batch: [1200/1382], lr:0.00020963, loss: 4.2476
21-03-09 09:56:11 | Epoch: 1, batch: [1300/1382], lr:0.0002271, loss: 4.7492
Valid | Epoch:1, loss:4.5008, training_time:9.8 min
21-03-09 09:57:52 | Epoch: 2, batch: [100/1382], lr:0.00025889, loss: 5.0385
21-03-09 09:58:33 | Epoch: 2, batch: [200/1382], lr:0.00027636, loss: 4.3349
21-03-09 09:59:15 | Epoch: 2, batch: [300/1382], lr:0.00029383, loss: 3.792
21-03-09 09:59:56 | Epoch: 2, batch: [400/1382], lr:0.0003113, loss: 4.09
21-03-09 10:00:37 | Epoch: 2, batch: [500/1382], lr:0.00032877, loss: 3.6593
21-03-09 10:01:18 | Epoch: 2, batch: [600/1382], lr:0.00034624, loss: 3.2719
21-03-09 10:01:59 | Epoch: 2, batch: [700/1382], lr:0.00036371, loss: 4.3402
21-03-09 10:02:40 | Epoch: 2, batch: [800/1382], lr:0.00038118, loss: 3.4255
21-03-09 10:03:21 | Epoch: 2, batch: [900/1382], lr:0.00039865, loss: 4.6646
21-03-09 10:04:02 | Epoch: 2, batch: [1000/1382], lr:0.00041612, loss: 3.6064
21-03-09 10:04:43 | Epoch: 2, batch: [1100/1382], lr:0.00043359, loss: 3.2388
21-03-09 10:05:24 | Epoch: 2, batch: [1200/1382], lr:0.00045106, loss: 2.9552
21-03-09 10:06:06 | Epoch: 2, batch: [1300/1382], lr:0.00046853, loss: 2.6404
Valid | Epoch:2, loss:3.1016, training_time:9.9 min
21-03-09 10:07:47 | Epoch: 3, batch: [100/1382], lr:0.00050032, loss: 2.5212
21-03-09 10:08:30 | Epoch: 3, batch: [200/1382], lr:0.00051779, loss: 3.0716
21-03-09 10:09:11 | Epoch: 3, batch: [300/1382], lr:0.00053526, loss: 4.1668
21-03-09 10:09:52 | Epoch: 3, batch: [400/1382], lr:0.00055273, loss: 4.8759
21-03-09 10:10:33 | Epoch: 3, batch: [500/1382], lr:0.0005702, loss: 3.0285
21-03-09 10:11:14 | Epoch: 3, batch: [600/1382], lr:0.00058767, loss: 2.6931
21-03-09 10:11:55 | Epoch: 3, batch: [700/1382], lr:0.00060514, loss: 2.8964
21-03-09 10:12:36 | Epoch: 3, batch: [800/1382], lr:0.00062261, loss: 2.5866
21-03-09 10:13:17 | Epoch: 3, batch: [900/1382], lr:0.00064007, loss: 2.6209
21-03-09 10:13:58 | Epoch: 3, batch: [1000/1382], lr:0.00065754, loss: 2.5504
21-03-09 10:14:39 | Epoch: 3, batch: [1100/1382], lr:0.00067501, loss: 2.6558
21-03-09 10:15:21 | Epoch: 3, batch: [1200/1382], lr:0.00069248, loss: 2.4132
21-03-09 10:16:02 | Epoch: 3, batch: [1300/1382], lr:0.00069325, loss: 2.3948
Valid | Epoch:3, loss:2.5894, training_time:9.9 min
21-03-09 10:17:43 | Epoch: 4, batch: [100/1382], lr:0.00067823, loss: 2.0928
21-03-09 10:18:24 | Epoch: 4, batch: [200/1382], lr:0.00067038, loss: 2.2669
21-03-09 10:19:05 | Epoch: 4, batch: [300/1382], lr:0.0006628, loss: 2.2461
21-03-09 10:19:46 | Epoch: 4, batch: [400/1382], lr:0.00065547, loss: 2.0225
21-03-09 10:20:27 | Epoch: 4, batch: [500/1382], lr:0.00064837, loss: 1.8909
21-03-09 10:21:08 | Epoch: 4, batch: [600/1382], lr:0.00064151, loss: 2.4791
21-03-09 10:21:50 | Epoch: 4, batch: [700/1382], lr:0.00063485, loss: 2.1045
21-03-09 10:22:31 | Epoch: 4, batch: [800/1382], lr:0.0006284, loss: 1.9807
21-03-09 10:23:12 | Epoch: 4, batch: [900/1382], lr:0.00062214, loss: 2.414
21-03-09 10:23:53 | Epoch: 4, batch: [1000/1382], lr:0.00061607, loss: 2.3508
21-03-09 10:24:34 | Epoch: 4, batch: [1100/1382], lr:0.00061017, loss: 3.2864
21-03-09 10:25:15 | Epoch: 4, batch: [1200/1382], lr:0.00060444, loss: 2.4069
21-03-09 10:25:57 | Epoch: 4, batch: [1300/1382], lr:0.00059886, loss: 2.2001
Valid | Epoch:4, loss:2.2931, training_time:9.9 min
21-03-09 10:27:39 | Epoch: 5, batch: [100/1382], lr:0.0005891, loss: 2.0385
21-03-09 10:28:20 | Epoch: 5, batch: [200/1382], lr:0.00058393, loss: 1.9714
21-03-09 10:29:01 | Epoch: 5, batch: [300/1382], lr:0.0005789, loss: 1.6864
21-03-09 10:29:43 | Epoch: 5, batch: [400/1382], lr:0.000574, loss: 1.71
21-03-09 10:30:23 | Epoch: 5, batch: [500/1382], lr:0.00056922, loss: 1.6959
21-03-09 10:31:05 | Epoch: 5, batch: [600/1382], lr:0.00056455, loss: 1.7439
21-03-09 10:31:46 | Epoch: 5, batch: [700/1382], lr:0.00056, loss: 1.7868
21-03-09 10:32:27 | Epoch: 5, batch: [800/1382], lr:0.00055556, loss: 2.0302
21-03-09 10:33:08 | Epoch: 5, batch: [900/1382], lr:0.00055122, loss: 1.6077
21-03-09 10:33:49 | Epoch: 5, batch: [1000/1382], lr:0.00054698, loss: 3.1334
21-03-09 10:34:30 | Epoch: 5, batch: [1100/1382], lr:0.00054284, loss: 1.8202
21-03-09 10:35:12 | Epoch: 5, batch: [1200/1382], lr:0.00053879, loss: 1.7087
21-03-09 10:35:53 | Epoch: 5, batch: [1300/1382], lr:0.00053483, loss: 2.0621
Valid | Epoch:5, loss:2.1809, training_time:9.9 min
21-03-09 10:37:34 | Epoch: 6, batch: [100/1382], lr:0.00052784, loss: 1.7812
21-03-09 10:38:15 | Epoch: 6, batch: [200/1382], lr:0.00052412, loss: 1.5168
21-03-09 10:38:57 | Epoch: 6, batch: [300/1382], lr:0.00052047, loss: 2.2757
21-03-09 10:39:38 | Epoch: 6, batch: [400/1382], lr:0.0005169, loss: 1.6701
21-03-09 10:40:19 | Epoch: 6, batch: [500/1382], lr:0.0005134, loss: 1.7323
21-03-09 10:41:01 | Epoch: 6, batch: [600/1382], lr:0.00050997, loss: 1.6406
21-03-09 10:41:42 | Epoch: 6, batch: [700/1382], lr:0.00050661, loss: 1.866
21-03-09 10:42:23 | Epoch: 6, batch: [800/1382], lr:0.00050331, loss: 1.7669
21-03-09 10:43:04 | Epoch: 6, batch: [900/1382], lr:0.00050008, loss: 1.5675
21-03-09 10:43:45 | Epoch: 6, batch: [1000/1382], lr:0.00049691, loss: 1.764
21-03-09 10:44:26 | Epoch: 6, batch: [1100/1382], lr:0.0004938, loss: 1.6566
21-03-09 10:45:07 | Epoch: 6, batch: [1200/1382], lr:0.00049074, loss: 1.8667
21-03-09 10:45:48 | Epoch: 6, batch: [1300/1382], lr:0.00048775, loss: 1.6022
Valid | Epoch:6, loss:2.1723, training_time:9.9 min
21-03-09 10:47:30 | Epoch: 7, batch: [100/1382], lr:0.00048243, loss: 1.17
21-03-09 10:48:11 | Epoch: 7, batch: [200/1382], lr:0.00047958, loss: 1.3194
21-03-09 10:48:52 | Epoch: 7, batch: [300/1382], lr:0.00047678, loss: 1.4456
21-03-09 10:49:33 | Epoch: 7, batch: [400/1382], lr:0.00047403, loss: 1.5089
21-03-09 10:50:15 | Epoch: 7, batch: [500/1382], lr:0.00047133, loss: 1.5307
21-03-09 10:50:56 | Epoch: 7, batch: [600/1382], lr:0.00046867, loss: 1.4507
21-03-09 10:51:37 | Epoch: 7, batch: [700/1382], lr:0.00046605, loss: 1.4995
21-03-09 10:52:18 | Epoch: 7, batch: [800/1382], lr:0.00046348, loss: 1.6506
21-03-09 10:52:59 | Epoch: 7, batch: [900/1382], lr:0.00046096, loss: 1.4541
21-03-09 10:53:41 | Epoch: 7, batch: [1000/1382], lr:0.00045847, loss: 1.5271
21-03-09 10:54:22 | Epoch: 7, batch: [1100/1382], lr:0.00045602, loss: 1.5686
21-03-09 10:55:03 | Epoch: 7, batch: [1200/1382], lr:0.00045361, loss: 1.5683
21-03-09 10:55:44 | Epoch: 7, batch: [1300/1382], lr:0.00045124, loss: 1.4195
Valid | Epoch:7, loss:2.1121, training_time:9.9 min
21-03-09 10:57:26 | Epoch: 8, batch: [100/1382], lr:0.00044702, loss: 1.2943
21-03-09 10:58:07 | Epoch: 8, batch: [200/1382], lr:0.00044475, loss: 1.3168
21-03-09 10:58:48 | Epoch: 8, batch: [300/1382], lr:0.00044252, loss: 1.2685
21-03-09 10:59:29 | Epoch: 8, batch: [400/1382], lr:0.00044032, loss: 1.3403
21-03-09 11:00:10 | Epoch: 8, batch: [500/1382], lr:0.00043815, loss: 1.3131
21-03-09 11:00:51 | Epoch: 8, batch: [600/1382], lr:0.00043601, loss: 1.4531
21-03-09 11:01:32 | Epoch: 8, batch: [700/1382], lr:0.0004339, loss: 1.1512
21-03-09 11:02:13 | Epoch: 8, batch: [800/1382], lr:0.00043183, loss: 1.4423
21-03-09 11:02:55 | Epoch: 8, batch: [900/1382], lr:0.00042978, loss: 1.3128
21-03-09 11:03:36 | Epoch: 8, batch: [1000/1382], lr:0.00042776, loss: 1.461
21-03-09 11:04:17 | Epoch: 8, batch: [1100/1382], lr:0.00042577, loss: 1.1839
21-03-09 11:04:58 | Epoch: 8, batch: [1200/1382], lr:0.00042381, loss: 1.3997
21-03-09 11:05:39 | Epoch: 8, batch: [1300/1382], lr:0.00042187, loss: 1.1995
Valid | Epoch:8, loss:2.1477, training_time:9.9 min
21-03-09 11:07:18 | Epoch: 9, batch: [100/1382], lr:0.00041842, loss: 1.0578
21-03-09 11:07:59 | Epoch: 9, batch: [200/1382], lr:0.00041656, loss: 1.2392
21-03-09 11:08:41 | Epoch: 9, batch: [300/1382], lr:0.00041472, loss: 1.1342
21-03-09 11:09:21 | Epoch: 9, batch: [400/1382], lr:0.0004129, loss: 1.2397
21-03-09 11:10:03 | Epoch: 9, batch: [500/1382], lr:0.00041111, loss: 1.0934
21-03-09 11:10:44 | Epoch: 9, batch: [600/1382], lr:0.00040935, loss: 1.2356
21-03-09 11:11:25 | Epoch: 9, batch: [700/1382], lr:0.0004076, loss: 1.1844
21-03-09 11:12:06 | Epoch: 9, batch: [800/1382], lr:0.00040588, loss: 1.2576
21-03-09 11:12:47 | Epoch: 9, batch: [900/1382], lr:0.00040418, loss: 1.1385
21-03-09 11:13:28 | Epoch: 9, batch: [1000/1382], lr:0.0004025, loss: 1.5052
21-03-09 11:14:10 | Epoch: 9, batch: [1100/1382], lr:0.00040084, loss: 1.1914
21-03-09 11:14:51 | Epoch: 9, batch: [1200/1382], lr:0.0003992, loss: 1.2752
21-03-09 11:15:32 | Epoch: 9, batch: [1300/1382], lr:0.00039758, loss: 1.2049
Valid | Epoch:9, loss:2.1855, training_time:9.8 min
```

```bash
python translate.py -dl_path /content/drive/MyDrive/zh_en.dl -ckpt_path /content/drive/MyDrive
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