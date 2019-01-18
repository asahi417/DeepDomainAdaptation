# Deep Domain Adaptation
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/WassersteinGAN/blob/master/LICENSE)

Tensorflow implementation of deep learning based domain adaptation models. 


## Get started

```
git clone https://github.com/asahi417/DeepDomainAdaptation
cd DeepDomainAdaptation
pip install .
```

## Script
### `bin/script_tfrecord.py`

Script to build tfrecord files.

```
usage: script_tfrecord.py [-h] --data [DATA]

This script is ...

optional arguments:
  -h, --help     show this help message and exit
  --data [DATA]  dataset name in dict_keys(['mnist', 'svhn'])
```

Environment variables:

| Environment variable name              | Default                                   | Description         |
| -------------------------------------- | ----------------------------------------- | ------------------- |
| **PATH_TO_CONFIG**                     | `./bin/config.json`                       | path to config file |


## List of Models
- Domain Adversarial Neural Network [[paper](https://arxiv.org/pdf/1505.07818.pdf), [implementation](./deep_da/model/dann.py)]
- Deep Joint Distribution Optimal Transport [[paper](https://arxiv.org/pdf/1803.10081.pdf), [implementation](./deep_da/model/jdot.py)]



