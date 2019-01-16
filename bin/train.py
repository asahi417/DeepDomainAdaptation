import domain_adaptation
import json
import toml
import os
import argparse


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='Model.', required=True, type=str, **share_param)
    parser.add_argument('--tar', help='Target.', default='mnist', type=str, **share_param)
    parser.add_argument('--src', help='Source.', default='svhn', type=str, **share_param)
    parser.add_argument('-e', '--epoch', help='Epoch.', required=True, type=int, **share_param)
    parser.add_argument('-v', '--version', help='number.', default=None, type=int, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # config
    config = json.load(open('./config.json'))

    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))

    ckpt_dir = '%s/%s' % (config['CKPT_DIR'], args.model)
    if args.version is None:
        param = toml.load(open('./bin/hyperparameter/%s.toml' % args.model))
        path_ckpt, _ = domain_adaptation.checkpoint_version(ckpt_dir, param)
    else:
        path_ckpt, param = domain_adaptation.checkpoint_version(ckpt_dir, version=args.version)

    path_to_tfrecord_target = '%s/%s' % (config['TFRECORD_DIR'], args.tar)
    path_to_tfrecord_source = '%s/%s' % (config['TFRECORD_DIR'], args.src)

    param['path_to_tfrecord_source'] = path_to_tfrecord_source
    param['path_to_tfrecord_target'] = path_to_tfrecord_target
    param['checkpoint_dir'] = path_ckpt

    if 'dann' in args.model:
        model = domain_adaptation.model.DANN(**param)
    elif 'source' in args.model:
        model = domain_adaptation.model.Source(**param)
    elif 'jdot' in args.model:
        model = domain_adaptation.model.JDOT(**param)
    else:
        raise ValueError('unknown model!')

    model.train(epoch=args.epoch)
