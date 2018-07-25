A Pytorch implementation of [QANet](https://arxiv.org/pdf/1804.09541.pdf)  

This repository is based on [NLPLearn/QANet](https://github.com/NLPLearn/QANet) and [marquezo/qanet-impl](https://github.com/marquezo/qanet-impl)

It can get **em: 70.155** and **f1: 79.432** peformance after 22 epochs(2730 batches per epoch) with EMA.

### Requirements
- PyTorch 0.4.0
- [torcheras](https://github.com/hackiey/torcheras)
- spacy
- tqdm

### Usage
#### Preprocess
    ```
    $ mkdir data
    $ python preprocess.py
    ```

#### Train
    ```
    $ mkdir log
    $ mkdir log/qanet
    $ python train.py 'some description'
    ```

#### Evaluate
    First set the log folder and epoch number in evaluate.py then execute the script.
    ```
    $ python evaluate.py
    ```