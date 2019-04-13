""" TFRecord modules

This script contains a class, which enables converting data to tfrecord format,
and gives an instance to be used in Tensorflow graph.
"""

import json
import tensorflow as tf
from time import time
from . import iterator
from ..util import create_log, raise_error, mkdir


VALID_DATA = dict(
    mnist=iterator.MNIST,
    svhn=iterator.SVHN
)


class TFRecorder:
    """ TFRecorder: conversion to TFRecord and utilize it in tensorflow graph

     Usage
    -----------
    - create TFRecord file
    >>> import deep_da
    >>> recorder = deep_da.TFRecorder()
    >>> recorder.create('mnist', dir_to_save='./', path_to_data=dict('path_to_mnist'))

    """

    def __init__(self):
        self.__logger = create_log()

    def create(self,
               dataset_name: str,
               dir_to_save: str,
               path_to_data: dict,
               progress_interval: int = 1):
        """ Create TFRecord fil

         Parameter
        ---------------------
        dataset_name: a name of dataset
        dir_to_save: path to directory where tfrecord files will be saved
        path_to_data: dictionary, consists of `train` and `valid` elements
        progress_interval: progress interval
        """

        def byte_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))

        mkdir(dir_to_save)
        raise_error(dataset_name not in VALID_DATA.keys(),
                    'unknown data %s not in %s' % (dataset_name, str(VALID_DATA.keys())))
        data_iterator_class = VALID_DATA[dataset_name]
        data_iterator = data_iterator_class(path_to_data=path_to_data)

        self.__logger.info('START CREATE TFRECORD:\n data: %s \n path: %s \n saved at: %s'
                           % (dataset_name, path_to_data, dir_to_save))

        for data_type in data_iterator.types:
            time_stamp_start = time()
            compress_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

            # set iterator type
            data_iterator.set_data_type(data_type)
            for data_label in [None] + data_iterator.data_label:
                data_iterator.set_data_label(data_label)
                data_label = '' if data_label is None else '_%i' % data_label
                path = '%s/%s%s.tfrecord' % (dir_to_save, data_type, data_label)

                with tf.python_io.TFRecordWriter(path, options=compress_opt) as writer:
                    self.__logger.info(' - process %s with label %s (total size: %i)'
                                       % (data_type, data_label, data_iterator.data_size))

                    for n, data in enumerate(data_iterator):
                        feature = dict(data=byte_feature(data['data']), label=byte_feature(data['label']))
                        ex = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(ex.SerializeToString())

                        # output progress
                        if n % progress_interval == 0:
                            progress_perc = n / data_iterator.data_size * 100
                            whole_time = time() - time_stamp_start
                            print('%d / %d (%0.1f %%), (%0.1f sec) \r'
                                  % (n, data_iterator.data_size, progress_perc, whole_time), end='', flush=True)

        self.__logger.info(' - closing iterator')
        data_iterator.close(dir_to_save)
        self.__logger.info('Completed :)')

    @staticmethod
    def read_tf(dir_to_tfrecord: str):
        """ Get instance to be used in tensorflow graph

         Parameter
        ---------------------
        dir_to_tfrecord: path to directory where tfrecord files are saved (same as `dir_to_save` when create files)
        """

        meta_dict = json.load(open('%s/meta.json' % dir_to_tfrecord))

        lookup_label = json.load(open('%s/lookup_label.json' % dir_to_tfrecord))
        lookup_label_inv = dict((i, k) for k, i in lookup_label.items())

        def __read_tf(example_proto):
            features = dict(
                data=tf.FixedLenFeature((), tf.string, default_value=""),
                label=tf.FixedLenFeature((), tf.string, default_value=""),
            )
            parsed_features = tf.parse_single_example(example_proto, features)

            feature_data = tf.decode_raw(parsed_features["data"], tf.int32)
            feature_data = tf.reshape(feature_data, meta_dict['data_shape'])

            feature_label = tf.decode_raw(parsed_features["label"], tf.int32)
            feature_label = tf.reshape(feature_label, [len(lookup_label)])

            return feature_data, feature_label

        meta_dict['lookup_label'] = lookup_label
        meta_dict['lookup_label_inv'] = lookup_label_inv
        return __read_tf, meta_dict

