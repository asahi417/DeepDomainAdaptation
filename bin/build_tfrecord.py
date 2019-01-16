""" Build TFRecord """


import json
import argparse
from domain_adaptation import TFRecorder


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('--embed', help='word embedding model.', default=None, type=str, **share_param)
    parser.add_argument('--data', help='dataset name.', required=True, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # config
    config = json.load(open('./config.json'))

    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))

    if args.data in ['mnist', 'svhn']:
        dir_to_save = '%s/%s' % (config["TFRECORD_DIR"], args.data)
    else:
        dir_to_save = '%s/%s/%s' % (config["TFRECORD_DIR"], args.data, args.embed)

    recorder = TFRecorder(dataset_name=args.data,
                          dir_to_save=dir_to_save,
                          path_to_data=config['DATASET_DIR'][args.data])

    # if args.data not in ['mnist', 'svhn']:
    #     if args.embed is None:
    #         raise ValueError('please provide word embedding')
    #     recorder.set_word_embedding(**config["WORD_EMBEDDINGS_PATH"]['en'][args.embed])
    
    recorder.create()
