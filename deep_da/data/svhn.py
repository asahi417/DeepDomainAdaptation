""" Iterator for SVHN data, supposed to be used in `deep_da/data/tfrecorder.py` """

import random
import numpy as np
import scipy.io
import os


# DEFAULT_ROOD_DIR = os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-2]), 'dataset/svhn')
DEFAULT_ROOD_DIR = os.path.join(os.path.expanduser("~"), 'deep_da')
CONFIG = {"train": "train_32x32.mat", "valid": "test_32x32.mat"}


def download_data(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    files = ['train_32x32.mat', 'test_32x32.mat']
    for _f in files:
        if not os.path.exists(os.path.join(dir_name, _f)):
            print('downloading %s data to %s ....' % (_f, dir_name))
            os.system('wget -O %s/%s http://ufldl.stanford.edu/housenumbers/%s' % (dir_name, _f, _f))


class SVHN:
    """ SVHN iterator
    - train_data: 60k data ([images, labels]) to train model
    - valid_data: 10k data ([images, labels]) for validation
    """

    def __init__(self,
                 root_dir: str=None,
                 batch: int=10):
        root_dir = root_dir if root_dir is not None else DEFAULT_ROOD_DIR
        self.__root_dir = os.path.join(root_dir, 'dataset', 'svhn')
        download_data(self.__root_dir)

        # training data
        image, label = self.image(os.path.join(self.__root_dir, CONFIG['train']))
        unique_label, count = np.unique(label, return_counts=True)
        self.__unique_label_size = len(unique_label)
        self.__training_data = dict([(i, image[label == i]) for i in unique_label])
        self.__training_data_count = dict([(i, len(image[label == i])) for i in unique_label])
        # validation data
        image, label = self.image(os.path.join(self.__root_dir, CONFIG['valid']))
        self.__validation_data = dict(image=image, label=label)

        self.__data_type = 'train'
        self.types = ['train', 'valid']
        self.batch = batch

    @property
    def data(self):
        return self.__training_data, self.__validation_data

    @staticmethod
    def image(filename):
        """Extract the images into a 4D tensor [image index, y, x, channels].
        Values are in range of [0, 255]. """
        mat = scipy.io.loadmat(filename)
        image = np.transpose(mat['X'], [3, 0, 1, 2])
        label = mat['y'][:, 0]
        label[label == 10] = 0
        return image, label

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
