import json
import numpy as np
import scipy.io


class SVHN:
    """
    Feeding SVHN data
    - train_data: 60k data ([images, labels]) to train model
    - valid_data: 10k data ([images, labels]) for validation
    """

    types = ['train', 'valid']
    __data_type = None

    def __init__(self,
                 path_to_data: str,
                 path_to_save: str = None):

        self.__path_to_data = path_to_data
        self.__path_to_save = path_to_save
        self.__data_size = dict(train=73257, valid=26032)
        self.__data = dict(
            train=self.image("%s/train_32x32.mat" % self.__path_to_data),
            valid=self.image("%s/test_32x32.mat" % self.__path_to_data)
        )
        self.__lookup_tag = dict([(i, i) for i in range(10)])

    @staticmethod
    def image(filename):
        """Extract the images into a 4D tensor [image index, y, x, channels].
        Values are in range of [0, 255].
        """
        mat = scipy.io.loadmat(filename)

        return dict(
            image=np.transpose(mat['X'], [3, 0, 1, 2]),
            tag=mat['y'][:, 0]
        )

    def set_data_type(self, data_type: str):
        self.__data_type = data_type

    @property
    def data_size(self):
        return self.__data_size[self.__data_type]

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
        tag_index = self.__data[self.__data_type]['tag'][self.__ind]
        tag_index = 0 if tag_index == 10 else tag_index

        # one hot
        tag = np.zeros(len(self.__lookup_tag))
        tag[self.__lookup_tag[tag_index]] = 1

        img = self.__data[self.__data_type]['image'][self.__ind]
        result = dict(
            image=img.astype(np.int32),
            tag=tag.astype(np.int32)
        )
        self.__ind += 1
        return result

    def close(self, dir_to_save):
        with open('%s/lookup_tag.json' % dir_to_save, 'w') as f:
            json.dump(self.__lookup_tag, f)

        with open('%s/meta.json' % dir_to_save, 'w') as f:
            meta_dict = dict(tag_size=len(self.__lookup_tag),
                             size=self.__data_size,
                             image_shape=(32, 32, 3))
            json.dump(meta_dict, f)
