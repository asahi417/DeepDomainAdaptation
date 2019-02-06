""" Domain Adversarial Neural Network

Tensorflow implementation of adversarial training based domain adaptation model described in:
    Ganin, Yaroslav, et al.
    "Domain-adversarial training of neural networks."
    The Journal of Machine Learning Research 17.1 (2016): 2096-2030.
"""

import os
import numpy as np
import tensorflow as tf
from .util.util_tf import variable_summaries, FlipGradientBuilder, StepScheduler, VALID_BASIC_CELL
from ..util import create_log
from ..data import TFRecorder
from .default_hyperparameter import Parameter


TFR = TFRecorder()


class DANN:
    """ Domain Adversarial Neural Network
    
     Usage
    -----------
    >>> import deep_da
    >>> model_instance = deep_da.model.DANN()
    >>> # train model
    >>> model_instance.train(epoch=10)

    """

    def __init__(self,
                 model_checkpoint_version: int = None,
                 root_dir: str = '.',
                 **kwargs):
        """ DANN (Domain Adversarial Neural Network) model

         Parameter
        -------------------
        checkpoint_dir: path to checkpoint directory
        regularizer_config_domain_classification: dictionary of scheduling configuration for domain classification regularizer
        regularizer_config_feature_extraction: dictionary of scheduling configuration for feature extraction regularizer
        learning_rate_config: dictionary of scheduling configuration for learning rate
        path_to_tfrecord_source: path to tfrecord file (source data)
        path_to_tfrecord_target: path to tfrecord file (target data)
        config_feature_extractor: dictionary of configuration for feature extractor
        config_domain_classifier: dictionary of configuration for domain classifier
        config_model: dictionary of configuration for model
        batch: batch size
        optimizer: optimizer ['adam', 'momentum', 'sgd']
        weight_decay: weight decay
        keep_prob: dropout keep probability
        n_thread: number of thread for tfrecord
        ckpt_epoch: checkpoint epoch for warm start
        initializer: initializer ['variance_scaling', 'truncated_normal']
        batch_for_test: batch size for validation or test
        is_image: if the data is image
        base_cell: base cell from ['cnn', 'fc']
        warm_start: if warm start
        """

        # setting  hyper parameters
        checkpoint_dir = os.path.join(root_dir, 'checkpoint')
        if model_checkpoint_version is None:
            param_instance = Parameter(
                'dann', checkpoint_dir=checkpoint_dir, custom_parameter=kwargs
            )
        else:
            param_instance = Parameter(
                'dann', checkpoint_dir=checkpoint_dir, model_checkpoint_version=model_checkpoint_version
            )
        self.__config_learning_rate = param_instance('config_learning_rate')
        self.__config_domain_classifier = param_instance('config_domain_classifier')
        self.__config_feature_extractor = param_instance('config_feature_extractor')
        self.__config_regularizer_domain_classifier = param_instance('config_regularizer_domain_classifier')
        self.__config_regularizer_feature_extractor = param_instance('config_regularizer_feature_extractor')
        self.__config_classifier = param_instance('config_classifier')
        self.__batch = param_instance('batch')
        self.__optimizer = param_instance('optimizer')
        self.__weight_decay = param_instance('weight_decay')
        self.__keep_prob = param_instance('keep_prob')
        self.__n_thread = param_instance('n_thread')
        self.__initializer = param_instance('initializer')
        self.__base_cell = param_instance('base_cell')
        self.__tfrecord_source = param_instance('tfrecord_source')
        self.__tfrecord_target = param_instance('tfrecord_target')
        self.__checkpoint_path = param_instance.checkpoint_path

        # tfrecorder
        self.__read_tf_src, self.__meta_src = TFR.read_tf(dir_to_tfrecord=self.__tfrecord_source)
        self.__read_tf_tar, self.__meta_tar = TFR.read_tf(dir_to_tfrecord=self.__tfrecord_target)
        self.__tfrecord_path = dict(
            source=dict(
                train=os.path.join(self.__tfrecord_source, 'train.tfrecord'),
                valid=os.path.join(self.__tfrecord_source, 'valid.tfrecord')
            ),
            target=dict(
                train=os.path.join(self.__tfrecord_target, 'train.tfrecord'),
                valid=os.path.join(self.__tfrecord_target, 'valid.tfrecord')
            )
        )

        # resize width and height to be fit smaller one (source and target)
        self.__resize = int(np.min(self.__meta_src['data_shape'][0:2] + self.__meta_tar['data_shape'][0:2]))

        # tile channel to be fit larger one (source and target)
        if self.__meta_src['data_shape'][-1] not in [1, 3] or self.__meta_tar['data_shape'][-1] not in [1, 3]:
            raise ValueError('invalid shape: tar %s, src %s'
                             % (self.__meta_tar['data_shape'], self.__meta_src['data_shape']))
        if self.__meta_tar['data_shape'][-1] == self.__meta_src['data_shape'][-1]:
            self.__tile_channel = None
        elif self.__meta_tar['data_shape'][-1] > self.__meta_src['data_shape'][-1]:
            self.__tile_channel = 'src'
        else:
            self.__tile_channel = 'tar'
        self.__channel = max(self.__meta_tar['data_shape'][-1], self.__meta_src['data_shape'][-1])

        # base model component configuration
        if self.__base_cell not in VALID_BASIC_CELL.keys():
            raise ValueError('invalid base_cell: %s not in %s' % (self.__base_cell, VALID_BASIC_CELL.keys()))
        base_model = VALID_BASIC_CELL[self.__base_cell]
        shape_input = [self.__resize, self.__resize, self.__channel]
        self.__feature_extractor = base_model.FeatureExtractor(shape_input, **self.__config_feature_extractor)
        self.__classifier = base_model.Model(**self.__config_classifier)
        self.__domain_classifier = base_model.DomainClassifier(**self.__config_domain_classifier)

        # create tensorflow graph
        self.__logger = create_log(os.path.join(self.__checkpoint_path, 'training.log'))
        self.__logger.info('BUILD DANN TENSORFLOW GRAPH')
        self.__build_graph()
        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.__writer = tf.summary.FileWriter('%s/summary' % self.__checkpoint_path, self.__session.graph)

        # load model
        if os.path.exists(os.path.join(self.__checkpoint_path, 'model.ckpt.meta')):
            self.__logger.info('load model from %s' % self.__checkpoint_path)
            self.__saver.restore(self.__session, os.path.join(self.__checkpoint_path, 'model.ckpt'))
            self.__warm_start = True
        else:
            self.__session.run(tf.global_variables_initializer())
            self.__warm_start = False
        
    def __tfrecord(self,
                   batch,
                   is_training,
                   is_source):
        """ Get tfrecord iterator and its initializer

         Parameter
        -----------------------
        batch: batch size, possibly tensor of integer
        is_training: boolean tensor
        is_source: boolean value

         Return
        -----------------------
        iterator, initializer
        """

        if is_source:
            tf_reader = self.__read_tf_src
            tfrecord = self.__tfrecord_path['source']
        else:
            tf_reader = self.__read_tf_tar
            tfrecord = self.__tfrecord_path['target']

        tfrecord_name = tf.where(is_training, tfrecord['train'], tfrecord['valid'])
        data_set_api = tf.data.TFRecordDataset(tfrecord_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(tf_reader, self.__n_thread)
        # set buffer size
        # buffer_size = tf.where(is_training, 10000 if is_source else 60000, 1000)
        buffer_size = 5000
        data_set_api = data_set_api.shuffle(buffer_size=tf.cast(buffer_size, tf.int64))
        data_set_api = data_set_api.batch(tf.cast(batch, tf.int64))
        # make iterator
        iterator = tf.data.Iterator.from_structure(data_set_api.output_types, data_set_api.output_shapes)
        iterator_ini = iterator.make_initializer(data_set_api)
        return iterator, iterator_ini

    def __build_graph(self):
        """ build tensorflow graph

        G_f : feature extractor
        G_dc: domain classifier
        G_m : main task-specific model

        (x_t, y_t) ~ target
        (x_s, y_s) ~ source

        loss = L_main{ G_m(G_f(x_s)), y_s } - r_dc * L_da{ G_m(G_f(x_s)), G_m(G_f(x_t)) | r_fe }

        * Note: r_dc is regularization term for domain classification, and r_fe is coefficient to scale amount of
                propagation by feature extractor
        """

        ##################
        # initialization #
        ##################
        if self.__initializer == 'variance_scaling':
            initializer = tf.contrib.layers.variance_scaling_initializer()
        elif self.__initializer == 'truncated_normal':
            initializer = tf.initializers.truncated_normal(stddev=0.02)
        else:
            raise ValueError('unknown initializer: %s' % self.__initializer)

        self.learning_rate = tf.placeholder_with_default(0.0, [], name='learning_rate')
        self.regularizer_domain_classification = tf.placeholder_with_default(
            0.0, [], name='regularizer_domain_classification')
        self.regularizer_feature_extraction = tf.placeholder_with_default(
            0.0, [], name='regularizer_feature_extraction')
        self.is_training = tf.placeholder_with_default(False, [])

        __keep_prob = tf.where(self.is_training, self.__keep_prob, 1.0)
        __weight_decay = tf.where(self.is_training, self.__weight_decay, 0.0)

        # TFRecord
        iterator_src, self.__iterator_ini_src = self.__tfrecord(self.__batch, self.is_training, is_source=True)
        iterator_tar, self.__iterator_ini_tar = self.__tfrecord(self.__batch, self.is_training, is_source=False)

        # get next input
        image_src, label_src = iterator_src.get_next()
        image_tar, label_tar = iterator_tar.get_next()

        ##############
        # preprocess #
        ##############

        # onehot vector of label
        label_src = tf.cast(label_src, tf.float32)
        label_tar = tf.cast(label_tar, tf.float32)

        # resizing data
        if self.__resize is not None:
            image_tar = tf.image.resize_image_with_crop_or_pad(image_tar, self.__resize, self.__resize)
            image_src = tf.image.resize_image_with_crop_or_pad(image_src, self.__resize, self.__resize)

        # tiling image for channel
        if self.__tile_channel == 'tar':
            image_tar = tf.tile(image_tar, [1, 1, 1, 3])
        elif self.__tile_channel == 'src':
            image_src = tf.tile(image_src, [1, 1, 1, 3])

        # make the channel in between [-1, 1]
        image_src = tf.cast(image_src, tf.float32) / 225 * 2 - 1
        image_tar = tf.cast(image_tar, tf.float32) / 225 * 2 - 1

        ###################
        # overall network #
        ###################

        # universal feature extraction
        with tf.variable_scope('feature_extraction', initializer=initializer):
            feature_src = self.__feature_extractor(image_src, __keep_prob)
            feature_tar = self.__feature_extractor(image_tar, reuse=True)

        # task-specific model
        with tf.variable_scope('classifier', initializer=initializer):
            pred_prob_src = self.__classifier(feature_src)
            pred_prob_tar = self.__classifier(feature_tar, reuse=True)
            # loss
            loss_model_src = - tf.reduce_mean(label_src * tf.log(pred_prob_src + 1e-6))
            loss_model_tar = - tf.reduce_mean(label_tar * tf.log(pred_prob_tar + 1e-6))

        # domain classification
        with tf.variable_scope('domain_classification', initializer=initializer):
            flip_grad = FlipGradientBuilder()
            pred_prob_domain_src = self.__domain_classifier(
                flip_grad(feature_src, scale=self.regularizer_feature_extraction))
            pred_prob_domain_tar = self.__domain_classifier(
                flip_grad(feature_tar, scale=self.regularizer_feature_extraction), reuse=True)

            # loss for domain classification and feature extractor: target (1), source (0)
            loss_domain_classification = - tf.reduce_mean(
                tf.concat([tf.log(pred_prob_domain_tar + 1e-6), tf.log(1 - pred_prob_domain_src + 1e-6)], axis=0)
            )

        ################
        # optimization #
        ################
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # optimizer
        if self.__optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.__optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        elif self.__optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        else:
            raise ValueError('unknown optimizer !!')

        # L2 weight decay
        if __weight_decay != 0.0:
            l2 = __weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables])
        else:
            l2 = 0.0

        # optimization
        total_loss = loss_model_src + self.regularizer_domain_classification * loss_domain_classification + l2
        gradient = tf.gradients(total_loss, trainable_variables)
        self.__train_op = optimizer.apply_gradients(zip(gradient, trainable_variables))

        # accuracy
        accuracy_src = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(label_src, axis=1), tf.argmax(pred_prob_src, axis=1)), tf.float32
            )
        )
        accuracy_tar = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(label_tar, axis=1), tf.argmax(pred_prob_tar, axis=1)), tf.float32
            )
        )

        domain_accuracy_tar = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.cast(
                    tf.logical_not(tf.less(pred_prob_domain_tar, 0.5)),
                    tf.float32),
                    1.0),
                tf.float32
            )
        )
        domain_accuracy_src = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.cast(
                    tf.less(pred_prob_domain_src, 0.5),
                    tf.float32),
                    1.0),
                tf.float32
            )
        )

        accuracy_domain = tf.reduce_mean([domain_accuracy_tar, domain_accuracy_src])

        # saver
        self.__saver = tf.train.Saver()

        ##################
        # scalar summary #
        ##################
        self.__summary_train = tf.summary.merge([
            tf.summary.scalar('train_meta_learning_rate', self.learning_rate),
            tf.summary.scalar('train_meta_r_domain_classification', self.regularizer_domain_classification),
            tf.summary.scalar('train_meta_r_feature_extraction', self.regularizer_feature_extraction),
            tf.summary.scalar('train_meta_keep_prob', __keep_prob),
            tf.summary.scalar('train_meta_weight_decay', __weight_decay),
            tf.summary.scalar('train_eval_loss_model_src', loss_model_src),
            tf.summary.scalar('train_eval_loss_model_tar', loss_model_tar),
            tf.summary.scalar('train_eval_loss_domain', loss_domain_classification),
            tf.summary.scalar('train_eval_accuracy_src', accuracy_src),
            tf.summary.scalar('train_eval_accuracy_tar', accuracy_tar),
            tf.summary.scalar('train_eval_accuracy_domain', accuracy_domain)
        ])

        self.__summary_valid = tf.summary.merge([
            tf.summary.scalar('valid_meta_keep_prob', __keep_prob),
            tf.summary.scalar('valid_meta_weight_decay', __weight_decay),
            tf.summary.scalar('valid_eval_loss_model_src', loss_model_src),
            tf.summary.scalar('valid_eval_loss_model_tar', loss_model_tar),
            tf.summary.scalar('valid_eval_loss_domain', loss_domain_classification),
            tf.summary.scalar('valid_eval_accuracy_src', accuracy_src),
            tf.summary.scalar('valid_eval_accuracy_tar', accuracy_tar),
            tf.summary.scalar('valid_eval_accuracy_domain', accuracy_domain)
        ])

        n_var = 0

        var_stat = []
        for var in trainable_variables:
            sh = var.get_shape().as_list()
            self.__logger.info('%s: %s' % (var.name, str(sh)))
            var_stat.extend(variable_summaries(var, var.name.split(':')[0]))
            n_var += np.prod(sh)

        self.__summary_train_var = tf.summary.merge(var_stat)
        self.__logger.info('total variables: %i' % n_var)

    def train(self, epoch: int):
        if self.__warm_start:
            meta = np.load(os.path.join(self.__checkpoint_path, 'meta.npz'))
            i_summary_train = int(meta['i_summary_train'])
            i_summary_valid = int(meta['i_summary_valid'])
            i_summary_train_var = int(meta['i_summary_train_var'])
            ini_epoch = int(meta['epoch'])
        else:
            ini_epoch, i_summary_train, i_summary_valid, i_summary_train_var = 0, 0, 0, 0

        self.__logger.info('checkpoint (%s), epoch (%i)' % (self.__checkpoint_path, epoch))
        scheduler_lr = StepScheduler(current_epoch=ini_epoch, **self.__config_learning_rate)
        scheduler_r_dc = StepScheduler(current_epoch=ini_epoch, **self.__config_regularizer_domain_classifier)
        scheduler_r_fe = StepScheduler(current_epoch=ini_epoch, **self.__config_regularizer_feature_extractor)
        e = -1

        try:

            for e in range(ini_epoch, ini_epoch+epoch):

                self.__logger.info('epoch %i/%i' % (e, ini_epoch+epoch))

                self.__logger.info(' - training')
                self.__session.run([self.__iterator_ini_src, self.__iterator_ini_tar],
                                   feed_dict={self.is_training: True})
                feed_train = {
                    self.is_training: True,
                    self.learning_rate: scheduler_lr(),
                    self.regularizer_domain_classification: scheduler_r_dc(),
                    self.regularizer_feature_extraction: scheduler_r_fe()
                }

                while True:
                    try:
                        summary_train, _ = self.__session.run([self.__summary_train, self.__train_op],
                                                              feed_dict=feed_train)
                        self.__writer.add_summary(summary_train, i_summary_train)  # write tensorboard writer
                        i_summary_train += 1  # time stamp for tf summary
                    except tf.errors.OutOfRangeError:
                        break

                self.__logger.info(' - validation')
                self.__session.run([self.__iterator_ini_src, self.__iterator_ini_tar],
                                   feed_dict={self.is_training: False})
                while True:
                    try:
                        summary_valid = self.__session.run(self.__summary_valid, feed_dict={self.is_training: False})
                        self.__writer.add_summary(summary_valid, i_summary_valid)  # write tensorboard writer
                        i_summary_valid += 1  # time stamp for tf summary
                    except tf.errors.OutOfRangeError:
                        break

                if e % 20 == 0:  # every 20 epoch, save statistics of weights
                    summary_train_var = self.__session.run(self.__summary_train_var, feed_dict={self.is_training: False})
                    self.__writer.add_summary(summary_train_var, i_summary_train_var)  # write tensorboard writer
                    i_summary_train_var += 1  # time stamp for tf summary

            self.__logger.info('Completed :)')

        except KeyboardInterrupt:
            self.__logger.info('KeyboardInterrupt :(')

        self.__logger.info('Save checkpoints......')
        self.__saver.save(self.__session, os.path.join(self.__checkpoint_path, 'model.ckpt'))

        np.savez(os.path.join(self.__checkpoint_path, 'meta.npz'),
                 epoch=e + 1,
                 i_summary_train=i_summary_train,
                 i_summary_valid=i_summary_valid,
                 i_summary_train_var=i_summary_train_var)







