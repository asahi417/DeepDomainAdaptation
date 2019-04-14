""" Script to train model

python bin/script_train.py -m deep_jdot -e 100
"""

import deep_da
import os
import argparse


def get_options():
    parser = argparse.ArgumentParser(description='Train model.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='Model name in %s' % deep_da.util.get_model_instance().keys(),
                        required=True, type=str, **share_param)
    parser.add_argument('-e', '--epoch', help='Epoch',
                        required=True, type=int, **share_param)
    parser.add_argument('-v', '--version', help='Checkpoint version if train from existing checkpoint',
                        default=None, type=int, **share_param)
    parser.add_argument('--root_dir', help='Root directory to store checkpoint and data',
                        default=None, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = get_options()
    model_instance = deep_da.util.get_model_instance(args.model)
    model = model_instance(model_checkpoint_version=args.version, root_dir=args.root_dir)
    model.train(epoch=args.epoch)
