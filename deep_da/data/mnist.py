""" Iterator for MNIST data, supposed to be used in `deep_da/data/tfrecorder.py` """

import numpy as np
import gzip
import os
import random

# DEFAULT_ROOD_DIR = os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-2]), 'dataset/mnist')
DEFAULT_ROOD_DIR = os.path.join(os.path.expanduser("~"), 'deep_da')
CONFIG = {
    "train": {"image": "train-images-idx3-ubyte.gz", "label": "train-labels-idx1-ubyte.gz"},
    "valid": {"image": "t10k-images-idx3-ubyte.gz", "label": "t10k-labels-idx1-ubyte.gz"}
}


def download_data(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    files = ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    for _f in files:
        if not os.path.exists(os.path.join(dir_name, _f)):
            print('downloading %s data to %s ....' % (_f, dir_name))
            os.system('wget -O %s/%s http://yann.lecun.com/exdb/mnist/%s' % (dir_name, _f, _f))


class MNIST:
    """ MNIST iterator:
    - train_data: 60k data ([images, labels]) to train model
    - valid_data: 10k data ([images, labels]) for validation
    """

    def __init__(self,
                 root_dir: str=None,
                 batch: int=10):
        root_dir = root_dir if root_dir is not None else DEFAULT_ROOD_DIR
        self.__root_dir = os.path.join(root_dir, 'dataset', 'mnist')
        download_data(self.__root_dir)

        # training data
        data = self.image(os.path.join(self.__root_dir, CONFIG['train']['image']), 60000)
        label = self.label(os.path.join(self.__root_dir, CONFIG['train']['label']), 60000)
        unique_label, count = np.unique(label, return_counts=True)
        self.__unique_label_size = len(unique_label)
        self.__training_data = dict([(i, data[label == i]) for i in unique_label])
        self.__training_data_count = dict([(i, len(data[label == i])) for i in unique_label])
        self.batch_num = int(np.min(list(self.__training_data_count.values())) / batch)

        # validation data
        self.__validation_data = dict(image=self.image(os.path.join(self.__root_dir, CONFIG['valid']['image']), 10000),
                                      label=self.label(os.path.join(self.__root_dir, CONFIG['valid']['label']), 10000))

        self.__data_type = 'train'
        self.types = ['train', 'valid']
        self.batch = batch

    @staticmethod
    def image(filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels].
        Values are in range of [0, 255]."""

        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(28 * 28 * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, 28, 28, 1)
            return data

    @staticmethod
    def label(filename, num_images):
        """Extract the labels into a vector label IDs."""
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels

    def set_data_type(self, data_type: str):
        self.__data_type = data_type

    def __iter__(self):
        self.__ind = 0
        return self

    def __next__(self):
        if self.__data_type == 'train':
            if self.__ind + self.batch >= np.min(list(self.__training_data_count.values())):
                # shuffle training data
                for k, v in self.__training_data.items():
                    random.shuffle(v)
                    self.__training_data[k] = v
                raise StopIteration
            image = []
            label = []
            for k, v in self.__training_data.items():
                image.append(v[self.__ind:self.__ind + self.batch])
                tmp = np.zeros((self.batch, self.__unique_label_size), dtype=np.int)
                tmp[:, k] = 1
                label.append(tmp)
            image = np.vstack(image).astype(np.int)
            label = np.vstack(label)
        elif self.__data_type == 'valid':
            if self.__ind + self.batch >= len(self.__validation_data['image']):
                raise StopIteration
            image = self.__validation_data['image'][self.__ind:self.__ind + self.batch].astype(np.int)
            _label = self.__validation_data['label'][self.__ind:self.__ind + self.batch]
            label = np.zeros((self.batch, self.__unique_label_size), dtype=np.int)
            for b, i in enumerate(_label):
                label[b, i] = 1
        else:
            raise ValueError('undefined datatype')
        assert len(image) == len(label)
        self.__ind += self.batch
        return image, label
