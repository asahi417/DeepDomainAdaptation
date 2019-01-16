import json
import numpy as np
import gzip


class MNIST:
    """
    Feeding MNIST data
    - train_data: 60k data ([images, labels]) to train model
    - valid_data: 10k data ([images, labels]) for validation
    """
    
    types = ['train', 'valid']
    __data_type = None

    def __init__(self,
                 path_to_data: str,
                 path_to_save: str=None):

        self.__path_to_data = path_to_data
        self.__path_to_save = path_to_save
        self.__data_size = dict(train=60000, valid=10000)
        self.__data = dict(
            train=dict(
                image=self.image("%s/train-images-idx3-ubyte.gz" % self.__path_to_data, 60000),
                tag=self.label("%s/train-labels-idx1-ubyte.gz" % self.__path_to_data, 60000)
            ),
            valid=dict(
                image=self.image("%s/t10k-images-idx3-ubyte.gz" % self.__path_to_data, 10000),
                tag=self.label("%s/t10k-labels-idx1-ubyte.gz" % self.__path_to_data, 10000)
            )
        )
        self.__lookup_tag = dict([(i, i) for i in range(10)])

    @staticmethod
    def image(filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels].
        Values are in range of [0, 255].
        """

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
        tag = np.zeros(len(self.__lookup_tag))
        tag[self.__lookup_tag[self.__data[self.__data_type]['tag'][self.__ind]]] = 1
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
                             image_shape=(28, 28, 1)
                             )
            json.dump(meta_dict, f)
