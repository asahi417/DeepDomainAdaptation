import numpy as np
import tensorflow as tf
from deep_da.model.util import util_tf

"""
Models used in DANN paper
"""


class Model:

    __base_n_hidden = [3072, 2048]

    def __init__(self,
                 output_size: int=10,
                 n_hidden: list=None):

        __n_hidden = n_hidden or self.__base_n_hidden
        self.__n_hidden = __n_hidden + [output_size]

    def __call__(self,
                 feature,
                 scope=None,
                 reuse=None):
        n_hidden = [feature.get_shape()[1]] + self.__n_hidden

        with tf.variable_scope(scope or "domain_classifier", reuse=reuse):
            def fc(_feature, _n):
                _feature = util_tf.full_connected(_feature, [n_hidden[_n], n_hidden[_n + 1]], scope='fc_%i' % _n)
                return _feature

            for i in range(3):
                feature = fc(feature, i)
                if i != 2:
                    feature = tf.nn.relu(feature)

            feature = tf.nn.softmax(feature)
        tf.assert_equal(feature.get_shape()[1], 10)
        return feature


class FeatureExtractor:

    __base_cnn_channel = [64, 64, 128]
    __base_cnn_filter = [5, 5, 5]
    __base_cnn_stride = [2, 2, 2]

    def __init__(self,
                 image_shape: list,
                 cnn_channel: list=None,
                 cnn_filter: list=None,
                 cnn_stride: list = None):

        assert len(image_shape) == 3

        self.__image_shape = image_shape
        __cnn_channel = cnn_channel or self.__base_cnn_channel
        self.__cnn_channel = [self.__image_shape[-1]] + __cnn_channel
        self.__cnn_filter = cnn_filter or self.__base_cnn_filter
        self.__cnn_stride = cnn_stride or self.__base_cnn_stride

    def __call__(self,
                 image,
                 keep_prob=None,
                 scope=None,
                 reuse=None):

        with tf.variable_scope(scope or "feature_extractor", reuse=reuse):

            def conv_pool(_feature, _n, _keep_prob=None):
                _shape = [self.__cnn_filter[_n], self.__cnn_filter[_n],
                          self.__cnn_channel[_n], self.__cnn_channel[_n+1]]
                _feature = util_tf.convolution(_feature,
                                               weight_shape=_shape,
                                               stride=[self.__cnn_stride[_n]] * 2,
                                               padding='SAME',
                                               scope='conv_%i' % _n)
                _feature = tf.nn.relu(_feature)
                _feature = tf.nn.max_pool(_feature,
                                          ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME')
                if _keep_prob is not None:
                    _feature = tf.nn.dropout(_feature, keep_prob=_keep_prob)
                return _feature

            feature = conv_pool(image, 0)
            feature = conv_pool(feature, 1, keep_prob)
            feature = conv_pool(feature, 2)

            def flatten(layer):
                _size = np.prod(layer.get_shape().as_list()[1:])
                return tf.reshape(layer, [-1, _size])

            feature = flatten(feature)
            return feature


class DomainClassifier:

    __base_n_hidden = [1024, 1024]

    def __init__(self,
                 output_size=1,
                 n_hidden: list=None):
        __n_hidden = n_hidden or self.__base_n_hidden
        self.__n_hidden = __n_hidden + [output_size]

    def __call__(self,
                 feature,
                 scope=None,
                 reuse=None):
        n_hidden = [feature.get_shape()[1]] + self.__n_hidden

        with tf.variable_scope(scope or "domain_classifier", reuse=reuse):
            def fc(_feature, _n):
                _feature = util_tf.full_connected(_feature, [n_hidden[_n], n_hidden[_n + 1]], scope='fc_%i' % _n)
                return _feature

            for i in range(3):
                feature = fc(feature, i)
                if i != 2:
                    feature = tf.nn.relu(feature)

        tf.assert_equal(feature.get_shape()[1], 1)
        return tf.nn.sigmoid(feature)

