import os
import logging
import json
import numpy as np
from glob import glob


def checkpoint_version(checkpoint_dir: str,
                       config: dict = None,
                       version: int = None):
    """ Checkpoint versioner:
    - return checkpoint dir, which has same hyper parameter (config)
    - return checkpoint dir, which corresponds to the version
    - return new directory
    :param config:
    :param checkpoint_dir: `./checkpoint/lam
    :param version: number of checkpoint
    :return:
    """

    if version is not None:
        checkpoints = glob('%s/v%i/hyperparameters.json' % (checkpoint_dir, version))
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
        for parameter_path in glob('%s/*/hyperparameters.json' % checkpoint_dir):
            # if not os.path.isdir(i):  # ignore files
            #     continue
            i = parameter_path.replace('/hyperparameters.json', '')
            json_dict = json.load(open(parameter_path))
            if config == json_dict:
                target_checkpoints.append(i)
        if len(target_checkpoints) == 1:
            return target_checkpoints[0], config
        elif len(target_checkpoints) == 0:
            new_checkpoint_id = len(glob('%s/*/hyperparameters.json' % checkpoint_dir))
            new_checkpoint_path = '%s/v%i' % (checkpoint_dir, new_checkpoint_id)
            os.makedirs(new_checkpoint_path, exist_ok=True)
            with open('%s/hyperparameters.json' % new_checkpoint_path, 'w') as outfile:
                json.dump(config, outfile)
            return new_checkpoint_path, config
        else:
            raise ValueError('Checkpoints are duplicated')


def create_log(out_file_path=None):
    """ Logging
        If `out_file_path` is None, only show in terminal
        or else save log file in `out_file_path`
    Usage
    -------------------
    logger.info(message)
    logger.error(error)
    """

    # handler to record log to a log file
    if out_file_path is not None:
        # if os.path.exists(out_file_path):
        #     os.remove(out_file_path)
        logger = logging.getLogger(out_file_path)

        if len(logger.handlers) > 1:  # if there are already handler, return it
            return logger
        else:
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s")

            handler = logging.FileHandler(out_file_path)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            logger_stream = logging.getLogger()
            # check if some stream handlers are already
            if len(logger_stream.handlers) > 0:
                return logger
            else:
                handler = logging.StreamHandler()
                handler.setFormatter(formatter)
                logger.addHandler(handler)

                return logger
    else:
        # handler to output
        handler = logging.StreamHandler()
        logger = logging.getLogger()

        if len(logger.handlers) > 0:  # if there are already handler, return it
            return logger
        else:  # in case of no, make new output handler
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            return logger


# class LearningRateScheduler:
#     """
#     Learning rate scheduler. Updates the learning rate when no significant improvement has been made in the last
#     `patience` steps.
#     """
#
#     def __init__(self,
#                  initial_learning_rate: float,
#                  method: str = None,
#                  learning_rate_multiplier: float=0.2,
#                  minimal_learning_rate: float=0.0,
#                  threshold_multiplier=1.0005,
#                  patience=20,
#                  ):
#         """
#         Parameters
#         ----------
#         method:
#             'epoch' (decay by epoch), 'loss' (decay by loss) if None -> no decay
#         minimal_learning_rate:
#             The training should stop when the learning rate is below `minimal_learning_rate`.
#         threshold_multiplier: (method 'loss')
#             Threshold multiplier used to check if learning rate should be updated or not.
#         patience: (method 'loss')
#             Lookback window to check if the learning rate should be updated or not.
#         """
#
#         self.method = method
#         self.initial_lr = initial_learning_rate
#         self.current_learning_rate = initial_learning_rate
#         self.learning_rate_multiplier = learning_rate_multiplier
#         self.minimal_learning_rate = minimal_learning_rate
#         self.stop = False
#
#         # parameter for method 'loss'
#         self.threshold_multiplier = threshold_multiplier
#         self.patience = patience
#         self.evaluations_since_last_decrease = 0
#         self.loss_sequence = []
#
#     def __call__(self, *args, **kwargs):
#         """
#         Returns
#         -------
#         (current_learning_rate, stop): (float, bool)
#             `current_learning_rate` is the updated learning rate which should be used in the next training step.
#             `stop` is a boolean which tells whether to stop training or not.
#         """
#         if self.method == 'epoch':
#             return self.__epoch()
#         elif self.method == 'loss':
#             return self.__loss(*args, **kwargs)
#         else:
#             return self.__identity()
#
#     def __epoch(self):
#         """Decay by epoch"""
#         self.current_learning_rate = self.current_learning_rate * self.learning_rate_multiplier
#         if self.current_learning_rate <= self.minimal_learning_rate:
#             self.stop = True
#         return self.current_learning_rate, self.stop
#
#     def __loss(self, loss):
#         """Decay by loss"""
#         self.loss_sequence.append(loss)
#
#         self.evaluations_since_last_decrease += 1
#         if self.evaluations_since_last_decrease > self.patience * 2:
#
#             current_mean = np.mean(self.loss_sequence[-self.patience:])
#             previous_mean = np.mean(self.loss_sequence[-self.patience * 2:-self.patience])
#             mean_not_decreased = current_mean >= previous_mean * self.threshold_multiplier
#
#             current_median = np.median(self.loss_sequence[-self.patience:])
#             previous_median = np.median(self.loss_sequence[-self.patience * 2:-self.patience])
#             median_not_decreased = current_median >= previous_median * self.threshold_multiplier
#
#             if mean_not_decreased and median_not_decreased:
#                 self.current_learning_rate *= self.learning_rate_multiplier
#                 self.evaluations_since_last_decrease = 0
#
#                 if self.current_learning_rate <= self.minimal_learning_rate:
#                     self.stop = True
#
#         return self.current_learning_rate, self.stop
#
#     def __identity(self):
#         return self.current_learning_rate, False