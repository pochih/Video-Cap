[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source-150x25.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

## Video-Captioning

<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='https://vsubhashini.github.io/imgs/S2VTarchitecture.png' padding='5px' height="250px"></img>
<a href='https://vsubhashini.github.io/imgs/S2VTarchitecture.png'>Image src</a>

- This repository is an implement of an ICCV2015 paper [Sequence to Sequence -- Video to Text](https://arxiv.org/abs/1505.00487) in Tensorflow 1.0

- Beside the basic model, I add attention mechanism to the original one. The attention mechanism is reference to an ICLR2015 paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

- The details was described in [__introduction.pdf__](https://github.com/brianhuang1019/Video-Captioning/blob/master/introduction.pdf).

## performance
|method|BLEU@1 score|
|---|---
|seq2seq*|0.28|
|seq2seq+attention**|0.3

*seq2seq is the reproduction of paper's model

**seq2seq+attention is my improvement to the paper's model

## run the code with pre-trained weights
```bash
./run.sh <testing_video ids> <testing_video features>
```

## train the code from scratch
```bash
./train.sh
```

## test the code
```bash
./test.sh
```

## Author
Po-Chih Huang / [@brianhuang1019](http://brianhuang1019.github.io/)
