import numpy as np
import tensorflow as tf
from deep_da.model.util import util_tf

"""
Models used in JDOT paper
"""


class Model:

    def __init__(self, output_size: int=10):

        self.__output_size = output_size

    def __call__(self,
                 feature,
                 scope=None,
                 reuse=None):
        n_current = feature.get_shape()[1]

        with tf.variable_scope(scope or "model", reuse=reuse):
            feature = util_tf.full_connected(feature, [n_current, self.__output_size], scope='fc')
            feature = tf.nn.softmax(feature)
        tf.assert_equal(feature.get_shape()[1], 10)
        return feature


class FeatureExtractor:

    __base_cnn_channel = [32, 32, 64, 64, 128, 128]
    __base_cnn_filter = [3] * len(__base_cnn_channel)
    __base_cnn_stride = [2] * len(__base_cnn_channel)

    def __init__(self,
                 image_shape: list,
                 cnn_channel: list=None,
                 cnn_filter: list=None,
                 cnn_stride: list=None,
                 fc_n_hidden: int=128):

        assert len(image_shape) == 3

        self.__image_shape = image_shape
        __cnn_channel = cnn_channel or self.__base_cnn_channel
        self.__cnn_channel = [self.__image_shape[-1]] + __cnn_channel
        self.__cnn_filter = cnn_filter or self.__base_cnn_filter
        self.__cnn_stride = cnn_stride or self.__base_cnn_stride
        self.__fc_n_hidden = fc_n_hidden

    def __call__(self,
                 image,
                 keep_prob=None,
                 scope=None,
                 reuse=None):

        with tf.variable_scope(scope or "feature_extractor", reuse=reuse):

            def conv_pool(_feature, _n):
                _shape = [self.__cnn_filter[_n], self.__cnn_filter[_n],
                          self.__cnn_channel[_n], self.__cnn_channel[_n+1]]
                _feature = util_tf.convolution(_feature,
                                               weight_shape=_shape,
                                               stride=[self.__cnn_stride[_n]] * 2,
                                               padding='SAME',
                                               scope='conv_%i' % _n)
                _feature = tf.nn.sigmoid(_feature)
                return _feature

            feature = image
            for i in range(len(self.__cnn_channel) - 1):
                feature = conv_pool(feature, i)

            def flatten(layer):
                _size = np.prod(layer.get_shape().as_list()[1:])
                return tf.reshape(layer, [-1, _size])

            feature = flatten(feature)
            if keep_prob is not None:
                feature = tf.nn.dropout(feature, keep_prob=keep_prob)

            feature = util_tf.full_connected(feature, [feature.get_shape()[1], self.__fc_n_hidden], scope='fc')
            return feature


class DomainClassifier:

    def __init__(self, output_size: int=1):
        self.__output_size = output_size

    def __call__(self,
                 feature,
                 scope=None,
                 reuse=None):

        n_current = feature.get_shape()[1]

        with tf.variable_scope(scope or "domain_classifier", reuse=reuse):
            feature = util_tf.full_connected(feature, [n_current, self.__output_size], scope='fc')
            feature = tf.nn.sigmoid(feature)

        return feature
