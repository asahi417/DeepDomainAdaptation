""" modules to manage parameters of a model """

import toml
import os
import json
from glob import glob


ABS_PATH_TO_THIS_SCRIPT = os.path.dirname(os.path.abspath(__file__))


def checkpoint_version(checkpoint_dir: str,
                       config: dict = None,
                       version: int = None):
    """ Checkpoint versioner: Either of `config` or `version` need to be specified (`config` has priority)

     Parameter
    ---------------------
    checkpoint_dir: directory where specific model's checkpoints are (will be) saved, eg) `checkpoint/cnn`
    config: parameter configuration to find same setting checkpoint
    version: number of checkpoint to warmstart from

     Return
    --------------------
    path_to_checkpoint, config

    - if there are no checkpoints, having same config as provided one, return new version
        eg) in case there are 'checkpoint/cnn/{v0,v1,v2}`, path_to_checkpoint = 'checkpoint/cnn/v3'
    - if there is a checkpoint, which has same config as provided one, return that version
        eg) in case there are 'checkpoint/cnn/{v0,v1,v2}`, and `v2` has same config, path_to_checkpoint = 'checkpoint/cnn/v2'
    - if `config` is None, `version` is required.
        eg) in case there are 'checkpoint/cnn/{v0,v1,v2}`, path_to_checkpoint = 'checkpoint/cnn/v0' if `version`=0
    """

    if version is None and config is None:
        raise ValueError('either of `version` or `config` is needed.')

    if version is not None:
        checkpoints = glob(os.path.join(checkpoint_dir, 'v%i' % version, 'hyperparameters.json'))
        if len(checkpoints) == 0:
            raise ValueError('No checkpoint: %s, %s' % (checkpoint_dir, version))
        elif len(checkpoints) > 1:
            raise ValueError('Multiple checkpoint found: %s, %s' % (checkpoint_dir, version))
        else:
            parameter = json.load(open(checkpoints[0]))
            target_checkpoints_dir = checkpoints[0].replace('/hyperparameters.json', '')
            return target_checkpoints_dir, parameter

    elif config is not None:
        # check if there are any checkpoints with same hyperparameters
        target_checkpoints = []
        for parameter_path in glob(os.path.join(checkpoint_dir, '*/hyperparameters.json')):
            # if not os.path.isdir(i):  # ignore files
            #     continue
            i = parameter_path.replace('/hyperparameters.json', '')
            json_dict = json.load(open(parameter_path))
            if config == json_dict:
                target_checkpoints.append(i)
        if len(target_checkpoints) == 1:
            return target_checkpoints[0], config
        elif len(target_checkpoints) == 0:
            new_checkpoint_id = len(glob(os.path.join(checkpoint_dir, '*/hyperparameters.json')))
            new_checkpoint_path = os.path.join(checkpoint_dir, 'v%i' % new_checkpoint_id)
            os.makedirs(new_checkpoint_path, exist_ok=True)
            with open(os.path.join(new_checkpoint_path, 'hyperparameters.json'), 'w') as outfile:
                json.dump(config, outfile)
            return new_checkpoint_path, config
        else:
            raise ValueError('Checkpoints are duplicated')


class Parameter:
    """ Parameter management class """

    def __init__(self,
                 model_name: str,
                 checkpoint_dir: str,
                 root_dir: str='.',
                 custom_parameter: dict=None,
                 model_checkpoint_version: int=None):

        if model_checkpoint_version is None and custom_parameter is None:
            raise ValueError('either of `model_checkpoint_version` or `custom_parameter` is needed.')

        self.__parameter = toml.load(open(os.path.join(ABS_PATH_TO_THIS_SCRIPT, "%s.toml" % model_name), "r"))
        self.__parameter['tfrecord_source'] = os.path.join(root_dir, self.__parameter['tfrecord_source'])
        self.__parameter['tfrecord_target'] = os.path.join(root_dir, self.__parameter['tfrecord_target'])
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)

        if model_checkpoint_version is None:
            full_parameter = dict()
            for k, v in self.__parameter.items():
                if k in custom_parameter.keys():
                    full_parameter[k] = custom_parameter[k]
                else:
                    full_parameter[k] = v
            self.__checkpoint, self.__custom_parameter = checkpoint_version(
                checkpoint_dir=checkpoint_dir, config=full_parameter)
        else:
            self.__checkpoint, self.__custom_parameter = checkpoint_version(
                checkpoint_dir=checkpoint_dir, version=model_checkpoint_version)

    @property
    def checkpoint_path(self):
        """ path to checkpoint file """
        return self.__checkpoint

    @property
    def full_parameter(self):
        return self.__custom_parameter

    def __call__(self, name: str):
        if name not in self.__parameter.keys():
            raise ValueError('unknown parameter: %s' % name)

        if name in self.__custom_parameter.keys():
            return self.__custom_parameter[name]
        else:
            return self.__parameter[name]


