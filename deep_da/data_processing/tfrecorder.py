import gensim
import os
import json
import tensorflow as tf
from time import time
from . import dataset_dependency
from ..util import create_log


def raise_error(condition, msg):
    """raising error with a condition"""
    if condition:
        raise ValueError(msg)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class TFRecorder:

    valid_dataset_name = ['yelp_original', 'yelp_csv', 'amazon_review', 'amazon_review_multi', 'mnist', 'svhn']
    word_embedding = None

    # for `load_statistics`
    image_shape = None
    lookup_label = None
    lookup_label_inv = None

    # lookup_char = None
    # lookup_char_inv = None
    # max_token_length = None
    # embedding_size = None
    # max_char_length = None
    # label_total_num = None
    # label_depth = None
    # max_mention_length = None

    def __init__(self,
                 dir_to_save: str,  # directory to save lookups eg) '~/tfrecord'
                 path_to_data: dict=None,  # dict(train='', valid='')
                 dataset_name: str=None,
                 debug: bool=True,
                 progress_interval: int=1,
                 is_image: bool=None):

        self.__name = dataset_name
        self.__dir_to_save = dir_to_save
        mkdir(self.__dir_to_save)
        self.__path_to_data = path_to_data

        self.__debug = debug
        self.__logger = create_log() if debug else None
        self.__progress_interval = progress_interval
        self.__log('dataset name: %s, dir: %s' % (self.__name, self.__dir_to_save))
        if is_image is None:
            self.__is_image = self.__name in ['mnist', 'svhn']
        else:
            self.__is_image = is_image

    def create(self):

        def byte_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))

        def int_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

        param = dict(path_to_data=self.__path_to_data,  path_to_save=self.__dir_to_save)
        if self.__name in ['mnist']:
            data_iterator = dataset_dependency.MNIST(**param)
        elif self.__name in ['svhn']:
            data_iterator = dataset_dependency.SVHN(**param)
        # elif
        #     embedding = self.word_embedding
        else:
            raise ValueError('invalid name: %s' % self.__name)

        for data_type in ['train', 'valid']:  # train, valid, test
            time_stamp_start = time()
            path = '%s/%s.tfrecord' % (self.__dir_to_save, data_type)
            compress_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

            # set iterator type
            data_iterator.set_data_type(data_type)

            with tf.python_io.TFRecordWriter(path, options=compress_opt) as writer:
                self.__log('process %s (total size: %i)' % (data_type, data_iterator.data_size))

                for n, data in enumerate(data_iterator):

                    # image
                    if self.__is_image:
                        feature = dict(image=byte_feature(data['image']), tag=byte_feature(data['tag']))

                    ex = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(ex.SerializeToString())

                    # output progress
                    if n % self.__progress_interval == 0:
                        progress_perc = n / data_iterator.data_size * 100
                        whole_time = time() - time_stamp_start
                        self.__print('%d / %d (%0.1f %%), (%0.1f sec) \r'
                                     % (n, data_iterator.data_size, progress_perc, whole_time), end='', flush=True)
                self.__print('')
                self.__log('finish processing.')
        self.__log('closing iterator')
        data_iterator.close(self.__dir_to_save)
        self.__log('Completed :) ')

    def load_statistics(self):
        meta_dict = json.load(open('%s/meta.json' % self.__dir_to_save))
        if self.__is_image:
            self.image_shape = meta_dict['image_shape']
        self.lookup_label = json.load(open('%s/lookup_tag.json' % self.__dir_to_save))
        self.lookup_label_inv = dict((i, k) for k, i in self.lookup_label.items())

    def read_tf(self):

        if self.lookup_label is None:
            self.load_statistics()

        if self.__is_image:
            def __read_tf(example_proto):
                features = dict(
                    image=tf.FixedLenFeature((), tf.string, default_value=""),
                    tag=tf.FixedLenFeature((), tf.string, default_value=""),
                )
                parsed_features = tf.parse_single_example(example_proto, features)

                feature_image = tf.decode_raw(parsed_features["image"], tf.int32)
                feature_image = tf.reshape(feature_image, self.image_shape)

                feature_tag = tf.decode_raw(parsed_features["tag"], tf.int32)
                feature_tag = tf.reshape(feature_tag, [len(self.lookup_label)])

                return feature_image, feature_tag
            return __read_tf



    # def read_tf(self, sentence_wise_option: bool):
    #     """reader supposed to be used in tensorflow graph"""
    #
    #     def __read_tf(examples):
    #         if self.lookup_char is None or self.lookup_label is None:
    #             self.load_statistics(sentence_wise_option)
    #
    #         features = dict(
    #             word=tf.FixedLenFeature((), tf.string, default_value=''),
    #             word_length=tf.FixedLenFeature((), tf.int64, default_value=0),
    #             char=tf.FixedLenFeature((), tf.string, default_value=''),
    #             char_length=tf.FixedLenFeature((), tf.string, default_value=''),
    #             position=tf.FixedLenFeature((), tf.string, default_value=''),
    #         )
    #
    #         if not sentence_wise_option:
    #             features['start_end'] = tf.FixedLenFeature((), tf.string, default_value='')
    #
    #         for i in range(self.lookup_label['depth'] + 1):
    #             if i == 0:
    #                 features['tag'] = tf.FixedLenFeature((), tf.string, default_value='')
    #             else:
    #                 features['tag_%i' % i] = tf.FixedLenFeature((), tf.string, default_value='')
    #
    #         parsed_features = tf.parse_single_example(examples, features)
    #
    #         def decode(name, shape, cast_type=tf.float32, raw_type=tf.float64):
    #             tmp = parsed_features[name]
    #             tmp = tf.decode_raw(tmp, raw_type)
    #             tmp = tf.cast(tmp, cast_type)
    #             tmp = tf.reshape(tmp, shape)
    #             return tmp
    #
    #         w_l = tf.cast(parsed_features['word_length'], tf.int32)
    #
    #         w_e = decode('word', [self.max_token_length, self.embedding_size])
    #         c_e = decode('char', [self.max_token_length, self.max_char_length], cast_type=tf.int32)
    #         c_l = decode('char_length', [self.max_token_length], cast_type=tf.int32)
    #         p = decode('position', [self.max_token_length, 5], cast_type=tf.int32)
    #
    #         labels = []
    #         for i in range(self.lookup_label['depth'] + 1):
    #             tag_name = 'tag' if i == 0 else 'tag_%i' % i
    #             if sentence_wise_option:
    #                 __shape = [self.max_token_length, self.label_total_num[tag_name]]
    #             else:
    #                 __shape = [self.label_total_num[tag_name]]
    #             labels.append(decode(tag_name, __shape, tf.int32))
    #
    #         if sentence_wise_option:
    #             return tuple([w_e, w_l, c_e, c_l, p] + labels)
    #         else:
    #             return tuple([w_e, w_l, c_e, c_l, p] + labels + [decode('start_end', [1, 2], cast_type=tf.int32)])
    #     return __read_tf
    #
    # def load_statistics(self, sentence_wise_option: bool):
    #     meta_dict = json.load(open('%s/meta.json' % self.__dir_to_save))
    #     self.max_token_length = meta_dict['max_token_length']
    #     self.max_mention_length = meta_dict['max_mention_length']
    #     self.embedding_size = meta_dict['embedding_size']
    #     self.max_char_length = meta_dict['max_char_length']
    #     self.label_total_num = meta_dict['label_total_num']
    #
    #     self.lookup_char = json.load(open('%s/lookup_char.json' % self.__dir_to_save))
    #     self.lookup_char_inv = dict((v, k) for k, v in self.lookup_char.items())
    #     self.lookup_label = json.load(open('%s/lookup_tag.json' % self.__dir_to_save))
    #
    #     # if sentence wise, add `unknown` tag to dict and redefine the whole tag number
    #     if sentence_wise_option:
    #         for keys in self.lookup_label.keys():
    #             if 'tag' in keys and 'cnt' not in keys:
    #                 self.label_total_num[keys] += 1
    #                 self.lookup_label[keys]['unknown'] = self.label_total_num[keys] - 1
    #
    #     self.lookup_label_inv = dict()
    #     for d in range(self.lookup_label['depth']+1):
    #         if d == 0:
    #             self.lookup_label_inv['tag'] = dict((v, k) for k, v in self.lookup_label['tag'].items())
    #         else:
    #             self.lookup_label_inv['tag_%i' % d] = dict((v, k) for k, v in self.lookup_label['tag_%i' % d].items())
    #     self.lookup_label_inv['position'] = dict((v, k) for k, v in self.lookup_label['position'].items())
    #     self.label_depth = self.lookup_label['depth']
    #
    # def set_word_embedding(self, *args, **kwargs):
    #     self.word_embedding = gensim.models.KeyedVectors.load_word2vec_format(*args, **kwargs)

    def __print(self, *args, **kwargs):
        if self.__debug:
            print(*args, **kwargs)

    def __log(self, msg):
        if self.__logger is not None:
            self.__logger.info(msg)