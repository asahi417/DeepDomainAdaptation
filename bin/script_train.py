""" Script to train model

python bin/script_train.py -m deep_jdot -e 100


Environment variables:

| Environment variable name | Default | Description         |
| ------------------------- | ------- | ------------------- |
| **ROOT_DIR**              | `.`     | root directory      |

"""

import deep_da
import os
import argparse

MODEL_LIST = dict(
    dann=deep_da.model.DANN,
    deep_jdot=deep_da.model.DeepJDOT
)
ROOT_DIR = os.getenv('ROOT_DIR', '.')


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='Model name in %s' % MODEL_LIST.keys(),
                        required=True, type=str, **share_param)
    parser.add_argument('-e', '--epoch', help='Epoch',
                        required=True, type=int, **share_param)
    parser.add_argument('-v', '--version', help='Checkpoint version if train from existing checkpoint',
                        default=None, type=int, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = get_options(
        argparse.ArgumentParser(description='Script to train models.',
                                formatter_class=argparse.RawTextHelpFormatter))

    if args.model not in MODEL_LIST.keys():
        raise ValueError('unknown model: %s not in %s' % (args.model, MODEL_LIST.keys()))

    model_instance = MODEL_LIST[args.model]
    model = model_instance(root_dir=ROOT_DIR, model_checkpoint_version=args.version)
    model.train(epoch=args.epoch)
