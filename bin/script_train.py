""" Script to train model

python bin/script_train.py -m deep_jdot -e 100
"""

import deep_da
import os
import argparse


PATH_TO_REPO = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])


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
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = get_options()

    list_of_model_name = deep_da.util.get_model_instance().keys()
    if args.model not in list_of_model_name:
        raise ValueError('unknown model: %s not in %s' % (args.model, list_of_model_name))

    model_instance = deep_da.util.get_model_instance(args.model)
    model = model_instance(root_dir=PATH_TO_REPO, model_checkpoint_version=args.version)
    model.train(epoch=args.epoch)
