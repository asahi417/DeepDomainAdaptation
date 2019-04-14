""" Domain Adversarial Neural Network

Tensorflow implementation of adversarial training based domain adaptation model described in:
    Ganin, Yaroslav, et al.
    "Domain-adversarial training of neural networks."
    The Journal of Machine Learning Research 17.1 (2016): 2096-2030.
"""

import os
import numpy as np
import tensorflow as tf
from .default_hyperparameter import Parameter
# from .util.util_tf import variable_summaries, FlipGradientBuilder, StepScheduler, VALID_BASIC_CELL
from . import util_tf
from ..util import create_log
from ..data import SVHN, MNIST


DEFAULT_ROOD_DIR = os.path.join(os.path.expanduser("~"), 'deep_da')


class DANN:
    """ Domain Adversarial Neural Network """

    def __init__(self,
                 root_dir: str = None,
                 **kwargs):

        # setting  hyper parameters
        root_dir = root_dir if root_dir is not None else DEFAULT_ROOD_DIR
        checkpoint_dir = os.path.join(root_dir, 'checkpoint')
        param_instance = Parameter('dann', checkpoint_dir=checkpoint_dir, custom_parameter=kwargs)
        self.__learning_rate = param_instance('learning_rate')
        self.__batch = param_instance('batch')
        self.__optimizer = param_instance('optimizer')
        self.__weight_decay = param_instance('weight_decay')
        self.__keep_prob = param_instance('keep_prob')
        self.__initializer = param_instance('initializer')
        self.__target_source = param_instance('target_source')

        if self.__target_source == 'mnist_svhn':
            self.__size_target = [None, 28, 28, 1]
            self.__size_source = [None, 32, 32, 3]
            self.iterator_target = MNIST(root_dir=root_dir, batch=self.__batch)
            self.iterator_source = SVHN(root_dir=root_dir, batch=self.__batch)
        elif self.__target_source == 'svhn_mnist':
            self.__size_target = [None, 32, 32, 3]
            self.__size_source = [None, 28, 28, 1]
            self.iterator_target = SVHN(batch=self.__batch)
            self.iterator_source = MNIST(batch=self.__batch)
        else:
            raise ValueError('undefined: %s' % self.__target_source)

        self.__regularizer_scheduler_bias = param_instance('regularizer_scheduler_bias')
        self.__checkpoint_path = param_instance.checkpoint_path

        # create tensorflow graph
        self.__logger = create_log(os.path.join(self.__checkpoint_path, 'log.log'))
        self.__logger.info('BUILD DANN TENSORFLOW GRAPH')
        self.__build_graph()
        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.__writer = tf.summary.FileWriter('%s/summary' % self.__checkpoint_path, self.__session.graph)
        self.__session.run(tf.global_variables_initializer())
        
    @staticmethod
    def __feature_extractor(image,
                            keep_prob=None,
                            scope=None,
                            reuse=None):

        out_cnn_channel = [64, 64, 128]
        in_cnn_channel = [image.get_shape().as_list()[-1], 64, 64]
        cnn_filter = [5, 5, 5]
        cnn_stride = [2, 2, 2]
        layer_n = len(out_cnn_channel)

        with tf.variable_scope(scope or "feature_extractor", reuse=reuse):
            # print(image.get_shape().as_list())
            for n in range(layer_n):
                shape = [cnn_filter[n], cnn_filter[n], in_cnn_channel[n], out_cnn_channel[n]]
                image = util_tf.convolution(
                    image, weight_shape=shape, stride=[cnn_stride[n]] * 2, padding='SAME', scope='conv_%i' % n)
                image = tf.nn.relu(image)
                if keep_prob is not None and n == 1:  # put dropout only second layer
                    image = tf.nn.dropout(image, keep_prob=keep_prob)
                if n != layer_n - 1:  # don't put max pool on last layer
                    image = tf.nn.max_pool(
                        image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                # print(image.get_shape().as_list())

            feature = tf.reshape(image, shape=[-1, np.prod(image.get_shape().as_list()[1:])])
            # print(feature.get_shape().as_list())
            return feature

    @staticmethod
    def __classifier(feature,
                     scope=None,
                     reuse=None):
        # out_n_hidden = [3072, 2048, 10]
        # in_n_hidden = [feature.get_shape().as_list()[-1], 3072, 2048]
        out_n_hidden = [64, 32, 10]
        in_n_hidden = [feature.get_shape().as_list()[-1], 64, 32]
        n_layer = len(out_n_hidden)

        with tf.variable_scope(scope or "classifier", reuse=reuse):
            for n in range(n_layer):
                feature = util_tf.full_connected(feature, [in_n_hidden[n], out_n_hidden[n]], scope='fc_%i' % n)
                if n != n_layer - 1:  # don't put relu on last layer
                    feature = tf.nn.relu(feature)
                else:
                    feature = tf.nn.softmax(feature)
        return feature

    @staticmethod
    def __domain_classifier(feature,
                            scope=None,
                            reuse=None):
        # out_n_hidden = [1024, 1024, 1]
        # in_n_hidden = [feature.get_shape().as_list()[-1], 1024, 1024]
        out_n_hidden = [64, 32, 1]
        in_n_hidden = [feature.get_shape().as_list()[-1], 64, 32]
        n_layer = len(out_n_hidden)

        with tf.variable_scope(scope or "domain_classifier", reuse=reuse):
            for n in range(n_layer):
                feature = util_tf.full_connected(feature, [in_n_hidden[n], out_n_hidden[n]], scope='fc_%i' % n)
                if n != n_layer - 1:  # don't put relu on last layer
                    feature = tf.nn.relu(feature)
                else:
                    feature = tf.nn.sigmoid(feature)
        return feature

    def __build_graph(self):
        """ build tensorflow graph """
        # configuration
        initializer = util_tf.get_initializer(self.__initializer)
        self.source_image = tf.placeholder(tf.float32, shape=self.__size_source, name='source_image')
        self.target_image = tf.placeholder(tf.float32, shape=self.__size_target, name='target_image')
        self.source_label = tf.placeholder(tf.float32, shape=[None, 10], name='source_label')
        self.target_label = tf.placeholder(tf.float32, shape=[None, 10], name='target_label')
        self.regularizer_feature_extraction = tf.placeholder(tf.float32, name='regularizer_feature_extraction')
        self.is_training = tf.placeholder_with_default(False, [])
        __keep_prob = tf.where(self.is_training, self.__keep_prob, 1.0)
        __weight_decay = tf.where(self.is_training, self.__weight_decay, 0.0)

        # normalize: make the channel in between [0, 1]
        # target_image = tf.image.per_image_standardization(self.target_image)
        # source_image = tf.image.per_image_standardization(self.source_image)
        target_image = self.target_image/255.0 - 1
        source_image = self.source_image/255.0 - 1

        # reshape and tile
        if self.__size_source[1] > self.__size_target[1]:  # source: svhn, target: mnist
            # source_image = tf.image.resize_bilinear(source_image, self.__size_target[1:3])
            source_image = tf.image.resize_image_with_crop_or_pad(
                source_image, self.__size_target[1], self.__size_target[2])
            target_image = tf.tile(target_image, [1, 1, 1, 3])
        else:
            # target_image = tf.image.resize_bilinear(target_image, self.__size_source[1:3])
            target_image = tf.image.resize_image_with_crop_or_pad(
                target_image, self.__size_source[1], self.__size_source[2])
            source_image = tf.tile(source_image, [1, 1, 1, 3])

        # shared feature extraction
        with tf.variable_scope('feature_extraction', initializer=initializer):
            source_feature = self.__feature_extractor(source_image, keep_prob=__keep_prob)
            target_feature = self.__feature_extractor(target_image, keep_prob=__keep_prob, reuse=True)

        # classifier
        with tf.variable_scope('classifier', initializer=initializer):
            source_prob = self.__classifier(source_feature)
            target_prob = self.__classifier(target_feature, reuse=True)

            # loss
            loss_source = - tf.reduce_mean(self.source_label * tf.log(source_prob + 1e-6))
            loss_target = - tf.reduce_mean(self.target_label * tf.log(target_prob + 1e-6))

        # domain classification
        with tf.variable_scope('domain_classification', initializer=initializer):
            flip_grad = util_tf.FlipGradientBuilder()
            source_domain_prob = self.__domain_classifier(
                flip_grad(source_feature, scale=self.regularizer_feature_extraction))
            target_domain_prob = self.__domain_classifier(
                flip_grad(target_feature, scale=self.regularizer_feature_extraction),
                reuse=True)

            # loss for domain classification and feature extractor: target (1), source (0)
            loss_source_domain = - tf.reduce_mean(tf.log(source_domain_prob + 1e-6))
            loss_target_domain = - tf.reduce_mean(tf.log(1 - target_domain_prob + 1e-6))
            loss_domain = tf.reduce_mean([loss_source_domain, loss_target_domain])

        # optimization
        with tf.variable_scope('loss'):
            total_loss = loss_source + loss_domain
            optimizer = util_tf.get_optimizer(self.__optimizer, self.__learning_rate)
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            # L2 weight decay
            if __weight_decay != 0.0:
                total_loss += __weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables])

            gradient = tf.gradients(total_loss, trainable_variables)
            gradient, _ = tf.clip_by_global_norm(gradient, 1)
            self.__train_op = optimizer.apply_gradients(zip(gradient, trainable_variables))

        # accuracy
        with tf.variable_scope('accuracy'):
            self.__accuracy_source = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(self.source_label, axis=1), tf.argmax(source_prob, axis=1)),
                    tf.float32))
            self.__accuracy_target = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(self.target_label, axis=1), tf.argmax(target_prob, axis=1)),
                    tf.float32))

            domain_accuracy_target = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.cast(
                        tf.logical_not(tf.less(source_domain_prob, 0.5)),
                        tf.float32),
                        1.0),
                    tf.float32))
            domain_accuracy_source = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.cast(
                        tf.less(target_domain_prob, 0.5),
                        tf.float32),
                        1.0),
                    tf.float32))
            accuracy_domain = tf.reduce_mean([domain_accuracy_target, domain_accuracy_source])

        # saver
        self.__saver = tf.train.Saver()

        # scalar summary
        self.__summary_train = tf.summary.merge([
            tf.summary.scalar('train_meta_r_feature_extraction', self.regularizer_feature_extraction),
            tf.summary.scalar('train_meta_keep_prob', __keep_prob),
            tf.summary.scalar('train_meta_weight_decay', __weight_decay),
            tf.summary.scalar('train_eval_loss_source', loss_source),
            tf.summary.scalar('train_eval_loss_target', loss_target),
            tf.summary.scalar('train_eval_loss_domain', loss_domain),
            tf.summary.scalar('train_eval_accuracy_source', self.__accuracy_source),
            tf.summary.scalar('train_eval_accuracy_target', self.__accuracy_target),
            tf.summary.scalar('train_eval_accuracy_domain', accuracy_domain)
        ])

        self.__summary_valid = tf.summary.merge([
            tf.summary.scalar('valid_meta_keep_prob', __keep_prob),
            tf.summary.scalar('valid_meta_weight_decay', __weight_decay),
            tf.summary.scalar('valid_eval_loss_source', loss_target),
            tf.summary.scalar('valid_eval_loss_target', loss_source),
            tf.summary.scalar('valid_eval_loss_domain', loss_domain),
            tf.summary.scalar('valid_eval_accuracy_source', self.__accuracy_source),
            tf.summary.scalar('valid_eval_accuracy_target', self.__accuracy_target),
            tf.summary.scalar('valid_eval_accuracy_domain', accuracy_domain)
        ])

        n_var = 0
        var_stat = []
        for var in trainable_variables:
            sh = var.get_shape().as_list()
            self.__logger.info('%s: %s' % (var.name, str(sh)))
            var_stat.extend(util_tf.variable_summaries(var, var.name.split(':')[0]))
            n_var += np.prod(sh)

        self.__summary_train_var = tf.summary.merge(var_stat)
        self.__logger.info('total variables: %i' % n_var)

    def regularizer_scheduler(self, total_epoch: int, data_size: int):
        # self.__regularizer_scheduler_bias = - 10
        total_step = total_epoch * data_size

        def scheduler(step, epoch):
            p = (step + epoch * data_size) / total_step
            alpha = 2. / (1. + np.exp(self.__regularizer_scheduler_bias * p)) - 1
            # alpha = min(alpha, 0.1)
            return alpha

        return scheduler

    def train(self, epoch: int):
        ini_epoch, i_summary_train, i_summary_valid, i_summary_train_var = 0, 0, 0, 0
        logger = create_log(os.path.join(self.__checkpoint_path, 'log_train.log'))
        logger.info('checkpoint (%s), epoch (%i)' % (self.__checkpoint_path, epoch))
        logger.info('- accuracy: source_train - target_train - source_valid - source_valid')
        batch_num = np.min([self.iterator_target.batch_num, self.iterator_source.batch_num])
        sample_in_epoch = int(batch_num * self.__batch * 10)
        scheduler_reg = self.regularizer_scheduler(epoch, sample_in_epoch)
        try:
            for e in range(ini_epoch, ini_epoch+epoch):

                # training
                self.iterator_source.set_data_type('train')
                self.iterator_target.set_data_type('train')
                it_source = iter(self.iterator_source)
                it_target = iter(self.iterator_target)
                accuracy_source, accuracy_target = [], []
                n = 0
                while True:
                    try:
                        image_source, label_source = next(it_source)
                        image_target, label_target = next(it_target)
                        feed_dict = {
                            self.source_image: image_source,
                            self.source_label: label_source,
                            self.target_image: image_target,
                            self.target_label: label_target,
                            self.regularizer_feature_extraction: scheduler_reg(n, e),
                            self.is_training: True
                        }
                        feed_val = [self.__summary_train, self.__accuracy_source, self.__accuracy_target, self.__train_op]
                        output = self.__session.run(feed_val, feed_dict=feed_dict)
                        self.__writer.add_summary(output[0], i_summary_train)  # write tensorboard writer
                        accuracy_source.append(output[1])
                        accuracy_target.append(output[2])
                        i_summary_train += 1  # time stamp for tf summary
                        n += 1
                    except StopIteration:
                        accuracy_source_train = float(np.mean(accuracy_source))
                        accuracy_target_train = float(np.mean(accuracy_target))
                        break

                # validation
                self.iterator_source.set_data_type('valid')
                self.iterator_target.set_data_type('valid')
                it_source = iter(self.iterator_source)
                it_target = iter(self.iterator_target)
                accuracy_source, accuracy_target = [], []
                while True:
                    try:
                        image_source, label_source = next(it_source)
                        image_target, label_target = next(it_target)
                        feed_dict = {
                            self.source_image: image_source,
                            self.source_label: label_source,
                            self.target_image: image_target,
                            self.target_label: label_target,
                            self.regularizer_feature_extraction: 0.0,
                            self.is_training: False
                        }
                        feed_val = [self.__summary_valid, self.__accuracy_source, self.__accuracy_target]
                        output = self.__session.run(feed_val, feed_dict=feed_dict)
                        self.__writer.add_summary(output[0], i_summary_valid)  # write tensorboard writer
                        accuracy_source.append(output[1])
                        accuracy_target.append(output[2])
                        i_summary_valid += 1  # time stamp for tf summary
                    except StopIteration:
                        accuracy_source_valid = float(np.mean(accuracy_source))
                        accuracy_target_valid = float(np.mean(accuracy_target))
                        break

                logger.info('epoch %i/%i: %0.2f, %0.2f, %0.2f, %0.2f (%0.4f)'
                            % (e, ini_epoch+epoch,
                               accuracy_source_train, accuracy_target_train, accuracy_source_valid, accuracy_target_valid,
                               scheduler_reg(n, e)))
                if e % 20 == 0:  # every 20 epoch, save statistics of weights
                    summary_train_var = self.__session.run(self.__summary_train_var, feed_dict={self.is_training: False})
                    self.__writer.add_summary(summary_train_var, i_summary_train_var)  # write tensorboard writer
                    i_summary_train_var += 1  # time stamp for tf summary

            logger.info('Completed :)')

        except KeyboardInterrupt:
            logger.info('KeyboardInterrupt :(')

        logger.info('Save checkpoints......')
        self.__saver.save(self.__session, os.path.join(self.__checkpoint_path, 'model.ckpt'))

