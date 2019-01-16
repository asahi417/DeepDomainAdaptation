"""
about batching --> https://stackoverflow.com/questions/44331612/how-to-set-a-number-for-epoch-in-tf-python-io-tf-record-iterator
about buffer size of shuffle --> https://github.com/tensorflow/tensorflow/issues/14857
pixel of each image is in range of [0 225]
"""

import os
import argparse
import json
import tensorflow as tf
from PIL import Image
import numpy as np
import domain_adaptation


OUTPUT = os.getenv('OUTPUT', './bin/check_tfrecord')

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT, exist_ok=True)


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-n', '--num', help='number.', default=10, type=int, **share_param)
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    # parser.add_argument('--train', help='Train dataset.', action='store_true')
    return parser.parse_args()


class TestTFRecord:

    def __init__(self,
                 path,
                 data,
                 n_thread=1):

        self.__n_thread = n_thread
        self.__recorder = domain_adaptation.TFRecorder(path, dataset_name=data, debug=False)
        self.__build_graph()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.session.run(tf.global_variables_initializer())
        self.session.run(self.data_iterator, feed_dict={self.tfrecord_name: '%s/train.tfrecord' % path})

    def __build_graph(self):

        ############
        # TFRecord #
        ############

        # load tfrecord instance
        self.tfrecord_name = tf.placeholder(tf.string, name='tfrecord_dataset_name')
        data_set_api = tf.data.TFRecordDataset(self.tfrecord_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(self.__recorder.read_tf(), self.__n_thread)
        # set batch size
        data_set_api = data_set_api.shuffle(buffer_size=10000, seed=0)
        data_set_api = data_set_api.batch(1)
        # make iterator
        iterator = tf.data.Iterator.from_structure(data_set_api.output_types, data_set_api.output_shapes)
        # get next input
        self.input_image, self.tag = iterator.get_next()

        # initialize iterator
        self.data_iterator = iterator.make_initializer(data_set_api)

    def get_data(self):
        img, tag = recorder.session.run([recorder.input_image, recorder.tag])
        img, tag = img[0], tag[0]
        tag = self.__recorder.lookup_label_inv[int(np.argmax(tag))]
        return img, int(tag)


if __name__ == '__main__':
    # config
    config = json.load(open('./config.json'))
    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))

    tfrecord_path = '%s/%s' % (config['TFRECORD_DIR'], args.data)

    recorder = TestTFRecord(tfrecord_path, args.data)

    for e in range(args.num):
        x, y = recorder.get_data()

        if args.data == 'svhn':
            x = np.rint(x).astype('uint8')
            x = Image.fromarray(x, 'RGB')
        elif args.data == 'mnist':
            x = np.rint(x[:, :, 0]).astype('uint8')
            x = Image.fromarray(x, 'L')
        else:
            raise ValueError('invalid shape')

        x.save('%s/%s-%i-n%s.png' % (OUTPUT, args.data, e, y))
