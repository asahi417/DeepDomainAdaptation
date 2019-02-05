""" Script to build TFRecord files

python bin/script_tfrecotd.py --data mnist
"""

import argparse
import os
import deep_da


PATH_TO_REPO = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])
CONFIG = {
    "mnist": {
      "train": {
        "image": os.path.join(PATH_TO_REPO, "dataset/mnist/train-images-idx3-ubyte.gz"),
        "label": os.path.join(PATH_TO_REPO, "dataset/mnist/train-labels-idx1-ubyte.gz")
      },
      "valid": {
        "image": os.path.join(PATH_TO_REPO, "dataset/mnist/t10k-images-idx3-ubyte.gz"),
        "label": os.path.join(PATH_TO_REPO, "dataset/mnist/t10k-labels-idx1-ubyte.gz")
      }
    },
    "svhn": {
      "train": os.path.join(PATH_TO_REPO, "dataset/svhn/train_32x32.mat"),
      "valid": os.path.join(PATH_TO_REPO, "dataset/svhn/test_32x32.mat")
    }
}


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('--data', help='dataset name in %s' % CONFIG.keys(), required=True, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options(
        argparse.ArgumentParser(
            description='This script converts dataset to tfrecord format.',
            formatter_class=argparse.RawTextHelpFormatter))

    if args.data not in CONFIG.keys():
        raise ValueError('unknown data name: %s not in %s' % (args.data, str(CONFIG.keys())))

    recorder = deep_da.TFRecorder()
    recorder.create(
        dataset_name=args.data,
        dir_to_save=os.path.join(PATH_TO_REPO, 'tfrecord/%s' % args.data),
        path_to_data=CONFIG[args.data]
    )
