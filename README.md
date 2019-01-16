# Deep Domain Adaptation
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/WassersteinGAN/blob/master/LICENSE)

Tensorflow implementation of deep learning based domain adaptation models. 


## Setup

```
git clone https://github.com/asahi417/DeepDomainAdaptation
cd DeepDomainAdaptation
pip install .
```

## List of Models
- Domain Adversarial Training [[paper](https://arxiv.org/pdf/1505.07818.pdf), [implementation](./deep_da/model/dann.py)]

## About Datasets (sentiment analysis dataset)
- Amazon Reviews
    - [Original](http://jmcauley.ucsd.edu/data/amazon/)(the original data requires documentations to download)
    - Without category
        - Used in sentiment analysis context ([Zhang et al, 2016](https://arxiv.org/pdf/1509.01626.pdf))
        - [download](https://github.com/zhangxiangxiao/Crepe/tree/master/data)
    - With category
        - Used in domain adaptation context ([Chen et al, 2012](https://arxiv.org/pdf/1206.4683.pdf))
        - [download](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/)
- Yelp data
    - [original](https://www.yelp.com/dataset)
    - [processed as csv](https://github.com/zhangxiangxiao/Crepe/tree/master/data)

