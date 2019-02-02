""" Iterator for SVHN data, supposed to be used in `deep_da/data/tfrecorder.py` """

import json
import numpy as np
import scipy.io


class SVHN:
    """ SVHN iterator
    - train_data: 60k data ([images, labels]) to train model
    - valid_data: 10k data ([images, labels]) for validation
    """

    def __init__(self, path_to_data: dict):

        self.__path_to_data = path_to_data
        self.__data_size = dict(train=73257, valid=26032)
        self.__data = dict(
            train=self.image(self.__path_to_data["train"]),
            valid=self.image(self.__path_to_data["valid"])
        )
        self.__lookup_label = dict([(i, i) for i in range(10)])
        self.types = ['train', 'valid']
        self.__data_type = None

    @staticmethod
    def image(filename):
        """Extract the images into a 4D tensor [image index, y, x, channels].
        Values are in range of [0, 255]. """
        mat = scipy.io.loadmat(filename)

        return dict(
            data=np.transpose(mat['X'], [3, 0, 1, 2]),
            label=mat['y'][:, 0]
        )

    @property
    def data_size(self):
        return self.__data_size[self.__data_type]

    def set_data_type(self, data_type: str):
        self.__data_type = data_type

    def __iter__(self):
        if self.__data_type is None or self.__data_type not in ['train', 'valid']:
            raise ValueError('set data type by `set_data_type`')
        self.__ind = 0
        return self

    def __next__(self):
        if self.__ind >= self.__data_size[self.__data_type]:
            raise StopIteration

        # raw data has index, in which digit `0` is indexed as 10 (other digits follow their number
        # eg `1`: 1, `2`: 2,..., `9`:9). So convert it to be `0` is indexed as 0.
        label_index = self.__data[self.__data_type]['label'][self.__ind]
        label_index = 0 if label_index == 10 else label_index

        # one hot
        label = np.zeros(len(self.__lookup_label))
        label[self.__lookup_label[label_index]] = 1

        img = self.__data[self.__data_type]['data'][self.__ind]
        result = dict(
            data=img.astype(np.int32),
            label=label.astype(np.int32)
        )
        self.__ind += 1
        return result

    def close(self, dir_to_save):
        with open('%s/lookup_label.json' % dir_to_save, 'w') as f:
            json.dump(self.__lookup_label, f)

        with open('%s/meta.json' % dir_to_save, 'w') as f:
            meta_dict = dict(
                label_size=len(self.__lookup_label),
                size=self.__data_size,
                data_shape=(32, 32, 3)
            )
            json.dump(meta_dict, f)
