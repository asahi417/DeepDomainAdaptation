""" Script to build TFRecord files

python bin/script_tfrecotd.py --data mnist


Environment variables:

| Environment variable name | Default | Description         |
| ------------------------- | ------- | ------------------- |
| **ROOT_DIR**              | `.`     | root directory      |

"""

import argparse
import os
import deep_da

ROOT_DIR = os.getenv('ROOT_DIR', '.')
CONFIG = {
    "mnist": {
      "train": {
        "image": os.path.join(ROOT_DIR, "dataset/mnist/train-images-idx3-ubyte.gz"),
        "label": os.path.join(ROOT_DIR, "dataset/mnist/train-labels-idx1-ubyte.gz")
      },
      "valid": {
        "image": os.path.join(ROOT_DIR, "dataset/mnist/t10k-images-idx3-ubyte.gz"),
        "label": os.path.join(ROOT_DIR, "dataset/mnist/t10k-labels-idx1-ubyte.gz")
      }
    },
    "svhn": {
      "train": os.path.join(ROOT_DIR, "dataset/svhn/train_32x32.mat"),
      "valid": os.path.join(ROOT_DIR, "dataset/svhn/test_32x32.mat")
    }
}


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('--data', help='dataset name in %s' % CONFIG['RAW_DATA'].keys(), required=True, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))

    if args.data not in CONFIG.keys():
        raise ValueError('unknown data name: %s not in %s' % (args.data, str(config['RAW_DATA'].keys())))

    recorder = deep_da.TFRecorder()
    recorder.create(
        dataset_name=args.data,
        dir_to_save=os.path.join(ROOT_DIR, 'tfrecord/%s' % args.data),
        path_to_data=CONFIG[args.data]
    )
