[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source-150x25.png?v=103)](https://github.com/ellerbrock/open-source-badges/)


## Video-Captioning

<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='https://vsubhashini.github.io/imgs/S2VTarchitecture.png' padding='5px' height="250px"></img>
<a href='https://vsubhashini.github.io/imgs/S2VTarchitecture.png'>Image src</a>

- This repository is an implement of an ICCV '15 paper [Sequence to Sequence -- Video to Text](https://arxiv.org/abs/1505.00487) in Tensorflow 1.0


## performance
|method|BLEU@1 score|
|---|---
|seq2seq*|0.28|

*seq2seq is the reproduction of paper's model

## run the code
```bash
pip install -r requirements.txt
```
```bash
./run.sh data/testing_id.txt data/test_features
```

for details, run.sh needs two parameters
```bash
./run.sh <video_id_file> <path_to_video_features>
```
- video_id_file

a txt file with video id

you can use [data/testing_id.txt](data/testing_id.txt) for convience

- path_to_video_features

a path contains video features, each video feature should be a *.npy file

take a look at [data/test_features](data/test_features)

you can use "data/test_features" directory for convience

## train the code
```bash
pip install -r requirements.txt
```
```bash
./train.sh
```

## test the code
```bash
./test.sh <path_to_model>
```
- path_to_model

the path to trained model

type "models/model-2380" to use pre-trained model

## Environment
- OS: CentOS Linux release 7.3.1611 (Core)
- CPU: Intel(R) Xeon(R) CPU E3-1230 v3 @ 3.30GHz
- GPU: GeForce GTX 1070 8GB
- Memory: 16GB DDR3
- Python3 (for data_parser.py) & Python2.7 (for others)

## Author
Po-Chih Huang / [@pochih](http://pochih.github.io/)
