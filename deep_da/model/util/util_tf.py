import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import numpy as np
import json
from glob import glob

from . import base_image_fc, base_image_cnn

VALID_BASIC_CELL = dict(
    cnn=base_image_cnn,
    fc=base_image_fc
)


def dynamic_batch_size(inputs):
    """ Dynamic batch size, which is able to use in a model without deterministic batch size.
    See https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn.py
    """
    while nest.is_sequence(inputs):
        inputs = inputs[0]
    return array_ops.shape(inputs)[0]


class StepScheduler:
    """ step size scheduler """

    def __init__(self,
                 current_epoch: int,
                 initial_step: float=None,
                 multiplier: float=1.0,
                 power: float=1.0,
                 exponential: bool=False,
                 identity: bool=False
                 ):

        if not exponential and initial_step is None:
            raise ValueError('initial step is needed')
        self.__initial_st = initial_step
        self.__current_ep = current_epoch
        self.__multiplier = multiplier
        self.__power = power
        self.__exponential = exponential
        self.__identity = identity

    def __call__(self):
        self.__current_ep += 1

        if self.__identity:
            return self.__initial_st
        elif self.__exponential:
            new_step = 2 / (1 + np.exp(-self.__multiplier * self.__current_ep)) - 1
            return new_step
        else:
            new_step = self.__initial_st / (1 + self.__multiplier * self.__current_ep) ** self.__power
            return new_step

    @property
    def initial_step(self):
        return self.__initial_st


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, scale=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * scale]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    https://www.tensorflow.org/guide/summaries_and_tensorboard """
    with tf.name_scope('var_%s' % name):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        return [tf.summary.scalar('mean', mean),
                tf.summary.scalar('stddev', stddev),
                tf.summary.scalar('max', tf.reduce_max(var)),
                tf.summary.scalar('min', tf.reduce_min(var)),
                tf.summary.histogram('histogram', var)]


def full_connected(x,
                   weight_shape,
                   scope=None,
                   bias=True,
                   reuse=None):
    """ fully connected layer
    - weight_shape: input size, output size
    - priority: batch norm (remove bias) > dropout and bias term
    """
    with tf.variable_scope(scope or "fully_connected", reuse=reuse):
        w = tf.get_variable("weight", shape=weight_shape, dtype=tf.float32)
        x = tf.matmul(x, w)
        if bias:
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
            return tf.add(x, b)
        else:
            return x


def convolution(x,
                weight_shape,
                stride,
                padding="SAME",
                scope=None,
                bias=True,
                reuse=None):
    """2d convolution
     Parameter
    -------------------
    weight_shape: width, height, input channel, output channel
    stride (list): [stride for axis 1, stride for axis 2]
    """
    with tf.variable_scope(scope or "2d_convolution", reuse=reuse):
        w = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32)
        x = tf.nn.conv2d(x, w, strides=[1, stride[0], stride[1], 1], padding=padding)
        if bias:
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
            return tf.add(x, b)
        else:
            return x


def dynamic_batch_size(inputs):
    """ Dynamic batch size, which is able to use in a model without deterministic batch size.
    See https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn.py
    """
    while nest.is_sequence(inputs):
        inputs = inputs[0]
    return array_ops.shape(inputs)[0]


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
