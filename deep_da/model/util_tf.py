import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops


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
