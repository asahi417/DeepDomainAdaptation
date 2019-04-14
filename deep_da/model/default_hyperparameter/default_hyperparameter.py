""" modules to manage parameters of a model """

import toml
import os
import json
from glob import glob


ABS_PATH_TO_THIS_SCRIPT = os.path.dirname(os.path.abspath(__file__))


class Parameter:
    """ Parameter management class """

    def __init__(self,
                 model_name: str,
                 checkpoint_dir: str,
                 custom_parameter: dict=None):

        self.__parameter = toml.load(open(os.path.join(ABS_PATH_TO_THIS_SCRIPT, "%s.toml" % model_name), "r"))
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)

        full_parameter = dict()
        for k, v in self.__parameter.items():
            if k in custom_parameter.keys():
                full_parameter[k] = custom_parameter[k]
            else:
                full_parameter[k] = v
        self.__checkpoint, self.__custom_parameter = self.checkpoint_version(
            checkpoint_dir=checkpoint_dir, config=full_parameter)

    @staticmethod
    def checkpoint_version(checkpoint_dir: str, config: dict):
        """ Checkpoint versioner: Either of `config` or `version` need to be specified (`config` has priority)

         Parameter
        ---------------------
        checkpoint_dir: directory where specific model's checkpoints are (will be) saved, eg) `checkpoint/cnn`
        config: parameter configuration to find same setting checkpoint

         Return
        --------------------
        path_to_checkpoint, config
        """

        # check if there are any checkpoints with same hyperparameters
        for parameter_path in glob(os.path.join(checkpoint_dir, '*/hyperparameters.json')):
            i = parameter_path.replace('/hyperparameters.json', '')
            json_dict = json.load(open(parameter_path))
            if config == json_dict:
                raise ValueError('checkpoints with same configuration exists: %s' % i)

        new_checkpoint_id = len(glob(os.path.join(checkpoint_dir, '*/hyperparameters.json')))
        new_checkpoint_path = os.path.join(checkpoint_dir, 'v%i' % new_checkpoint_id)
        os.makedirs(new_checkpoint_path, exist_ok=True)
        with open(os.path.join(new_checkpoint_path, 'hyperparameters.json'), 'w') as outfile:
            json.dump(config, outfile)
        return new_checkpoint_path, config

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


