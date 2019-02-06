# Deep Domain Adaptation
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/WassersteinGAN/blob/master/LICENSE)

Tensorflow implementation of deep learning based domain adaptation models. 
See more in [implemented algorithm](https://github.com/asahi417/DeepDomainAdaptation#list-of-models).

**Work in progress:** Experiment results will be appeared soon! 


## Get started

```
git clone https://github.com/asahi417/DeepDomainAdaptation
cd DeepDomainAdaptation
pip install .
```

- **install error**  
You might have install error if your environment dosen't have `numpy` and `cython`, due to the `pot` library.
Then, import them before install this repo.

```
pip install numpy
pip install cython
pip install .
```

## Script
### [`bin/script_tfrecord.py`](bin/script_tfrecord.py)

This script converts dataset to tfrecord format.

```
usage: script_tfrecord.py [-h] --data [DATA]

optional arguments:
  -h, --help     show this help message and exit
  --data [DATA]  dataset name in dict_keys(['mnist', 'svhn'])
```

### [`bin/script_train.py`](bin/script_train.py)

This script is to train models.

```
usage: script_train.py [-h] -m [MODEL] -e [EPOCH] [-v [VERSION]]

optional arguments:
  -h, --help            show this help message and exit
  -m [MODEL], --model [MODEL]
                        Model name in dict_keys(['dann', 'deep_jdot'])
  -e [EPOCH], --epoch [EPOCH]
                        Epoch
  -v [VERSION], --version [VERSION]
                        Checkpoint version if train from existing checkpoint
```

## List of Models
- Domain Adversarial Neural Network 
    - [Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." The Journal of Machine Learning Research 17.1 (2016): 2096-2030.](https://arxiv.org/pdf/1505.07818.pdf)
    - [implementation](./deep_da/model/dann.py)
- Deep Joint Distribution Optimal Transport
    - [Damodaran, Bharath Bhushan, et al. "DeepJDOT: Deep Joint distribution optimal transport for unsupervised domain adaptation." arXiv preprint arXiv:1803.10081 (2018).](https://arxiv.org/pdf/1803.10081.pdf)
    - [Nicolas Courty, et al. "Joint distribution optimal transportation for domain adaptation" Advances in Neural Information Processing Systems. 2017.](https://arxiv.org/pdf/1705.08848.pdf)
    - [implementation](./deep_da/model/deep_jdot.py)


