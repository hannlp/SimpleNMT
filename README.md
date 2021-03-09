# SimpleNMT
A simple and readable **Neural Machine Translation** system

## 1 background
* The process of **automatic translation of natural language by a computer** is called **Machine Translation (MT)**.
* **Neural Machine Translation (NMT)** directly uses the Encoder-Decoder framework to perform end-to-end mapping of Distributed Representation language, which has the advantages of unified model structure and high translation quality, and has become the mainstream of the times.
* The development of machine translation is mainly attributed to the promotion of **open source systems** and **evaluation competitions**. There are many excellent neural machine translation systems ([fairseq](https://github.com/pytorch/fairseq), [OpenNMT](https://opennmt.net/), [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), etc.), but these open source systems have the disadvantages of complex implementation, too much redundant code, and difficult for beginners to read.

## 2 To do
I am committed to building a Neural Machine Translation system that is easy to read, use, and friendly to beginners.

## 3 Documents
### 3.1 Download
```bash
git clone https://github.com/hannlp/SimpleNMT
cd 'SimpleNMT/simplenmt'
```

### 3.2 Train your model
```bash
python train.py -data_path .. -dl_path .. -ckpt_path ..
```