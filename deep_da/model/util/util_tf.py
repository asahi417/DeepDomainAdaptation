import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import numpy as np

from . import base_image_fc
from . import base_image_cnn

VALID_BASIC_CELL = dict(
    image=dict(
        cnn=base_image_cnn, fc=base_image_fc
    ),
    text='TBA'
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
