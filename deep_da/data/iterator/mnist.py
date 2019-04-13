""" Iterator for MNIST data, supposed to be used in `deep_da/data/tfrecorder.py` """

import json
import numpy as np
import gzip


class MNIST:
    """ MNIST iterator:
    - train_data: 60k data ([images, labels]) to train model
    - valid_data: 10k data ([images, labels]) for validation
    """

    def __init__(self, path_to_data: dict):

        self.__path_to_data = path_to_data
        self.__data_size = dict(train=60000, valid=10000)
        self.__data = dict(
            train=dict(
                data=self.image(self.__path_to_data['train']['image'], 60000),
                label=self.label(self.__path_to_data['train']['label'], 60000)
            ),
            valid=dict(
                data=self.image(self.__path_to_data['valid']['image'], 10000),
                label=self.label(self.__path_to_data['valid']['label'], 10000)
            )
        )
        self.__lookup_label = dict([(i, i) for i in range(10)])
        self.types = ['train', 'valid']
        self.__data_type = None
        self.__data_label = None

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

    @property
    def data_label(self):
        return list(self.__lookup_label.keys())

    @staticmethod
    def label(filename, num_images):
        """Extract the labels into a vector label IDs."""
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels

    @property
    def data_size(self):
        return self.__data_size[self.__data_type]

    def set_data_type(self, data_type: str):
        self.__data_type = data_type

    def set_data_label(self, data_label: int):
        if data_label is not None and data_label not in self.__lookup_label.keys():
            raise ValueError('unknown label %i' % data_label)
        self.__data_label = data_label

    def __iter__(self):
        if self.__data_type is None or self.__data_type not in ['train', 'valid']:
            raise ValueError('set data type by `set_data_type`')
        self.__ind = 0
        return self

    def __next__(self):
        while True:
            if self.__ind >= self.__data_size[self.__data_type]:
                raise StopIteration
            label = np.zeros(len(self.__lookup_label))
            label_id = self.__data[self.__data_type]['label'][self.__ind]
            if self.__data_label is None or self.__data_label == label_id:
                # one hot
                label[self.__lookup_label[label_id]] = 1
                img = self.__data[self.__data_type]['data'][self.__ind]
                result = dict(
                    data=img.astype(np.int32),
                    label=label.astype(np.int32)
                )
                self.__ind += 1
                break
            else:
                self.__ind += 1

        return result

    def close(self, dir_to_save):
        with open('%s/lookup_label.json' % dir_to_save, 'w') as f:
            json.dump(self.__lookup_label, f)

        with open('%s/meta.json' % dir_to_save, 'w') as f:
            meta_dict = dict(
                label_size=len(self.__lookup_label),
                size=self.__data_size,
                data_shape=(28, 28, 1)
            )
            json.dump(meta_dict, f)
