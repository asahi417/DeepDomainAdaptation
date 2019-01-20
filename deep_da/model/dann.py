""" Domain Adversarial Neural Network

Tensorflow implementation of adversarial training based domain adaptation model described in:
    Ganin, Yaroslav, et al.
    "Domain-adversarial training of neural networks."
    The Journal of Machine Learning Research 17.1 (2016): 2096-2030.
"""

import os
import numpy as np
import tensorflow as tf
from .util_tf import variable_summaries, FlipGradientBuilder, StepScheduler, VALID_BASIC_CELL
from ..util import create_log, raise_error
from ..data import TFRecorder


TFR = TFRecorder()


class DANN:
    """ Domain Adversarial Neural Network
    
     Usage
    -----------
    >>> import deep_da
    >>> model_instance = deep_da.model.DANN(**parameter)
    >>> # train model
    >>> model_instance.train(epoch=10)

    """

    def __init__(self,
                 checkpoint_dir: str,
                 regularizer_config_domain_classification: dict,
                 regularizer_config_feature_extraction: dict,
                 learning_rate_config: dict,
                 path_to_tfrecord_source: str,
                 path_to_tfrecord_target: str,
                 config_feature_extractor: dict=None,
                 config_domain_classifier: dict=None,
                 config_model: dict=None,
                 batch: int = 10,
                 optimizer: str = 'sgd',
                 weight_decay: float = 0.0,
                 keep_prob: float = 1.0,
                 n_thread: int = 4,
                 ckpt_epoch: int = None,
                 initializer: str='variance_scaling',
                 batch_for_test: int=1000,
                 is_image: bool = True,
                 base_cell: str = 'cnn',
                 warm_start: bool = True):
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

        self.__is_image = is_image

        # tfrecorder
        self.__read_tf_src, self.__meta_src = TFR.read_tf(dir_to_tfrecord=path_to_tfrecord_source, is_image=is_image)
        self.__read_tf_tar, self.__meta_tar = TFR.read_tf(dir_to_tfrecord=path_to_tfrecord_target, is_image=is_image)
        self.__path_to_tfrecord = dict(
            source=dict(
                train='%s/train.tfrecord' % path_to_tfrecord_source,
                valid='%s/valid.tfrecord' % path_to_tfrecord_source
            ),
            target=dict(
                train='%s/train.tfrecord' % path_to_tfrecord_target,
                valid='%s/valid.tfrecord' % path_to_tfrecord_target
            )
        )

        # basic structure
        if is_image:
            raise_error(base_cell not in VALID_BASIC_CELL['image'].keys(),
                        'invalid base_cell: %s not in %s' % (base_cell, VALID_BASIC_CELL['image'].keys()))
            base_model = VALID_BASIC_CELL['image'][base_cell]
            
            # resize width and height to be fit smaller one (source and target)
            self.__resize = int(np.min(self.__meta_src["image_shape"][0:2] + self.__meta_tar["image_shape"][0:2]))

            # tile channel to be fit larger one (source and target)
            raise_error(self.__meta_src["image_shape"][-1] not in [1, 3] or self.__meta_tar["image_shape"][-1] not in [1, 3],
                        'invalid shape: tar %s, src %s' % (self.__meta_tar["image_shape"], self.__meta_src["image_shape"]))
            if self.__meta_tar["image_shape"][-1] == self.__meta_src["image_shape"][-1]:
                self.__tile_channel = None
            elif self.__meta_tar["image_shape"][-1] > self.__meta_src["image_shape"][-1]:
                self.__tile_channel = 'src'
            else:
                self.__tile_channel = 'tar'
            self.__channel = max(self.__meta_tar["image_shape"][-1], self.__meta_src["image_shape"][-1]) 

            # model configuration
            self.__feature_extractor = base_model.FeatureExtractor([self.__resize, self.__resize, self.__channel],
                                                                   **config_feature_extractor)
            self.__domain_classifier = base_model.DomainClassifier(**config_domain_classifier)
            self.__model = base_model.Model(**config_model)
        else:
            raise ValueError('TBA')
            
        # checkpoint
        if ckpt_epoch is None:
            self.__checkpoint = '%s/model.ckpt' % checkpoint_dir
        else:
            self.__checkpoint = '%s/model-%i.ckpt' % (checkpoint_dir, ckpt_epoch)

        # training parameters
        self.__batch = batch
        self.__optimizer = optimizer
        self.__weight_decay = weight_decay
        self.__keep_prob = keep_prob
        self.__n_thread = n_thread
        self.__initializer = initializer
        self.__batch_for_test = batch_for_test

        self.__lr_config = learning_rate_config
        self.__reg_config_dc = regularizer_config_domain_classification
        self.__reg_config_fe = regularizer_config_feature_extraction

        self.__logger = create_log('%s/log' % checkpoint_dir)

        self.__logger.info('BUILD DANN GRAPH')
        self.__build_graph()

        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.__writer = tf.summary.FileWriter('%s/summary' % checkpoint_dir, self.__session.graph)

        # load model
        if os.path.exists('%s.meta' % self.__checkpoint) and warm_start:
            self.__logger.info('load variable from %s' % self.__checkpoint)
            self.__saver.restore(self.__session, self.__checkpoint)
            self.__warm_start = True
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
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
            path_to_tfrecord = self.__path_to_tfrecord['source']
        else:
            tf_reader = self.__read_tf_tar
            path_to_tfrecord = self.__path_to_tfrecord['target']

        tfrecord_name = tf.where(is_training, path_to_tfrecord['train'], path_to_tfrecord['valid'])
        batch = tf.where(is_training, batch, self.__batch_for_test)
        data_set_api = tf.data.TFRecordDataset(tfrecord_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(tf_reader, self.__n_thread)
        # set batch size
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
        image_src, tag_src = iterator_src.get_next()
        image_tar, tag_tar = iterator_tar.get_next()

        ##############
        # preprocess #
        ##############

        tag_src = tf.cast(tag_src, tf.float32)
        tag_tar = tf.cast(tag_tar, tf.float32)

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
        with tf.variable_scope('model', initializer=initializer):
            pred_prob_src = self.__model(feature_src)
            pred_prob_tar = self.__model(feature_tar, reuse=True)
            # loss
            loss_model_src = - tf.reduce_mean(tag_src * tf.log(pred_prob_src + 1e-6))
            loss_model_tar = - tf.reduce_mean(tag_tar * tf.log(pred_prob_tar + 1e-6))

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
        self.__accuracy_src = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(tag_src, axis=1), tf.argmax(pred_prob_src, axis=1)), tf.float32
            )
        )
        self.__accuracy_tar = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(tag_tar, axis=1), tf.argmax(pred_prob_tar, axis=1)), tf.float32
            )
        )

        acc_tar = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.cast(
                    tf.logical_not(tf.less(pred_prob_domain_tar, 0.5)),
                    tf.float32),
                    1.0),
                tf.float32
            )
        )
        acc_src = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.cast(
                    tf.less(pred_prob_domain_src, 0.5),
                    tf.float32),
                    1.0),
                tf.float32
            )
        )

        self.__accuracy_domain = tf.reduce_mean([acc_tar, acc_src])

        # saver
        self.__saver = tf.train.Saver()

        ##################
        # scalar summary #
        ##################
        self.__summary_train = tf.summary.merge([
            tf.summary.scalar('meta_learning_rate', self.learning_rate),
            tf.summary.scalar('meta_r_domain_classification', self.regularizer_domain_classification),
            tf.summary.scalar('meta_r_feature_extraction', self.regularizer_feature_extraction),
            tf.summary.scalar('meta_keep_prob', __keep_prob),
            tf.summary.scalar('meta_weight_decay', __weight_decay),
            tf.summary.scalar('eval_train_loss_model_src', loss_model_src),
            tf.summary.scalar('eval_train_loss_model_tar', loss_model_tar),
            tf.summary.scalar('eval_train_loss_domain', loss_domain_classification),
            tf.summary.scalar('eval_train_accuracy_src', self.__accuracy_src),
            tf.summary.scalar('eval_train_accuracy_tar', self.__accuracy_tar),
            tf.summary.scalar('eval_train_accuracy_domain', self.__accuracy_domain)
        ])

        self.__summary_valid = tf.summary.merge([
            tf.summary.scalar('eval_valid_loss_model_src', loss_model_src),
            tf.summary.scalar('eval_valid_loss_model_tar', loss_model_tar),
            tf.summary.scalar('eval_valid_loss_domain', loss_domain_classification),
            tf.summary.scalar('eval_valid_accuracy_src', self.__accuracy_src),
            tf.summary.scalar('eval_valid_accuracy_tar', self.__accuracy_tar),
            tf.summary.scalar('eval_valid_accuracy_domain', self.__accuracy_domain)
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
            dir_to_save = '/'.join(self.__checkpoint.split('/')[:-1])
            meta = np.load('%s/meta.npz' % dir_to_save)
            i_summary_train = int(meta['i_summary_train'])
            i_summary_valid = int(meta['i_summary_valid'])
            i_summary_train_var = int(meta['i_summary_train_var'])
            ini_epoch = int(meta['epoch'])
        else:
            ini_epoch, i_summary_train, i_summary_valid, i_summary_train_var = 0, 0, 0, 0

        self.__logger.info('checkpoint (%s), epoch (%i)' % (self.__checkpoint, epoch))
        scheduler_lr = StepScheduler(current_epoch=ini_epoch, **self.__lr_config)
        scheduler_r_dc = StepScheduler(current_epoch=ini_epoch, **self.__reg_config_dc)
        scheduler_r_fe = StepScheduler(current_epoch=ini_epoch, **self.__reg_config_fe)
        e = -1

        for e in range(ini_epoch, ini_epoch+epoch):

            self.__logger.info('epoch %i/%i' % (e, ini_epoch+epoch))

            # Train
            self.__session.run([self.__iterator_ini_src, self.__iterator_ini_tar], feed_dict={self.is_training: True})
            feed_train = {
                self.is_training: True,
                self.learning_rate: scheduler_lr(),
                self.regularizer_domain_classification: scheduler_r_dc(),
                self.regularizer_feature_extraction: scheduler_r_fe()
            }

            while True:
                try:
                    summary_train, _ = self.__session.run([self.__summary_train, self.__train_op], feed_dict=feed_train)
                    self.__writer.add_summary(summary_train, i_summary_train)  # write tensorboard writer
                    i_summary_train += 1  # time stamp for tf summary
                except tf.errors.OutOfRangeError:
                    break

            # validation
            self.__session.run([self.__iterator_ini_src, self.__iterator_ini_tar], feed_dict={self.is_training: False})
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

        self.__logger.info('FINISH SUCCESSFULLY :)')
        self.__saver.save(self.__session, self.__checkpoint)

        dir_to_save = '/'.join(self.__checkpoint.split('/')[:-1])
        np.savez('%s/meta.npz' % dir_to_save,
                 epoch=e+1,
                 i_summary_train=i_summary_train,
                 i_summary_train_var=i_summary_train_var,
                 i_summary_valid=i_summary_valid)








