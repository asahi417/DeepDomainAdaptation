""" Deep Joint Distribution Optimal Transport

Tensorflow implementation of Deep Joint Distribution Optimal Transport model described in:
    Damodaran, Bharath Bhushan, et al.
    "DeepJDOT: Deep Joint distribution optimal transport for unsupervised domain adaptation."
    arXiv preprint arXiv:1803.10081 (2018).
"""

import os
import numpy as np
import tensorflow as tf
from .util_tf import variable_summaries, FlipGradientBuilder, StepScheduler, VALID_BASIC_CELL
from ..util import create_log, raise_error
from ..data import TFRecorder


class DeepJDOT:
    """ Deep Joint Distribution Optimal Transport

     Usage
    -----------
    >>> import deep_da
    >>> model_instance = deep_da.model.DeepJDOT(**parameter)
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
                 config_feature_extractor: dict = None,
                 config_domain_classifier: dict = None,
                 config_model: dict = None,
                 batch: int = 10,
                 optimizer: str = 'sgd',
                 weight_decay: float = 0.0,
                 keep_prob: float = 1.0,
                 n_thread: int = 4,
                 ckpt_epoch: int = None,
                 initializer: str = 'variance_scaling',
                 batch_for_test: int = 1000,
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
            raise_error(
                self.__meta_src["image_shape"][-1] not in [1, 3] or self.__meta_tar["image_shape"][-1] not in [1, 3],
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
        # initializer
        if self.__initializer == 'variance_scaling':
            initializer = tf.contrib.layers.variance_scaling_initializer()
        elif self.__initializer == 'truncated_normal':
            initializer = tf.initializers.truncated_normal(stddev=0.02)
        else:
            raise ValueError('unknown initializer: %s' % self.__initializer)

        self.learning_rate = tf.placeholder_with_default(0.0, [], name='learning_rate')
        self.is_training = tf.placeholder_with_default(False, [])
        self.seed = tf.placeholder_with_default(0, [])

        __keep_prob = tf.where(self.is_training, self.__keep_prob, 1.0)
        __weight_decay = tf.where(self.is_training, self.__weight_decay, 0.0)

        # TFRecord
        iterator_src, self.__iterator_ini_src = self.__tfrecord(
            self.__batch, self.is_training, is_source=True, seed=self.seed
        )

        iterator_tar, self.__iterator_ini_tar = self.__tfrecord(
            self.__batch, self.is_training, is_source=False
        )

        # get next input
        image_tar, tag_tar = iterator_tar.get_next()
        tag_tar = tf.cast(tag_tar, tf.float32)

        image_src, tag_src = iterator_src.get_next()
        tag_src = tf.cast(tag_src, tf.float32)

        with tf.variable_scope('classifier', initializer=initializer):
            ##############
            # preprocess #
            ##############
            tag_src = tf.cast(tag_src, tf.float32)

            # resizing data
            if self.__resize is not None:
                image_tar = tf.image.resize_image_with_crop_or_pad(image_tar, self.__resize, self.__resize)
            # tiling image for channel
            if self.__tile_channel:
                image_tar = tf.tile(image_tar, [1, 1, 1, 3])

            # make the channel in between [-1, 1]
            image_src = tf.cast(image_src, tf.float32) / 225 * 2 - 1
            image_tar = tf.cast(image_tar, tf.float32) / 225 * 2 - 1

            ###################
            # overall network #
            ###################
            # universal feature extraction (`embedding` in context of deep JDOT)
            embedding_src = self.__feature_extractor(image_src, __keep_prob)
            embedding_tar = self.__feature_extractor(image_tar, reuse=True)

            embed_src = tf.expand_dims(embedding_src, 0)  # 1, batch, dim
            embed_src = tf.tile(embed_src, [self.__batch, 1, 1])  # batch, batch, dim

            embed_tar = tf.expand_dims(embedding_tar, 0)  # 1, batch, dim
            embed_tar = tf.transpose(embed_tar, [1, 0, 2])  # batch, 1, dim
            embed_tar = tf.tile(embed_tar, [1, self.__batch, 1])  # batch, batch, dim

            diff = embed_src - embed_tar

            # embedding distance: L2 loss (batch, batch)
            embedding_distance = tf.reduce_sum(diff * diff, axis=-1)

            # classifier
            pred_prob_src = self.__model(embedding_src)
            pred_prob_tar = self.__model(embedding_tar, reuse=True)

            pp_src = tf.expand_dims(tag_src, 0)  # 1, batch, dim
            pp_src = tf.tile(pp_src, [self.__batch, 1, 1])  # batch, batch, dim

            pp_tar = tf.expand_dims(pred_prob_tar, 0)  # 1, batch, dim
            pp_tar = tf.transpose(pp_tar, [1, 0, 2])  # batch, 1, dim
            pp_tar = tf.tile(pp_tar, [1, self.__batch, 1])  # batch, batch, dim

            # output distance: cross entropy (batch, batch)
            output_distance = tf.reduce_mean(pp_src * tf.log(pp_tar + 1e-6), axis=-1)

            # joint distance
            joint_distance = \
                self.__distance_coefficient_output * output_distance \
                + self.__distance_coefficient_embedding * embedding_distance

            # output distance (source only)
            output_distance_src = tf.reduce_mean(tag_src * tf.log(pred_prob_src + 1e-6), axis=1)

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

        with tf.variable_scope('optimal_transport', initializer=initializer):
            # transport matrix is constrained on simplex (sum of each element must be 1)
            tm = tf.get_variable("transport_matrix", shape=[self.__batch * self.__batch], dtype=tf.float32)
            tm = tf.nn.softmax(tm)
            tm = tf.reshape(tm, [self.__batch, self.__batch])
            transport = tf.reduce_sum(tm * joint_distance)

        ################
        # optimization #
        ################
        self.__loss_ot = transport
        self.__loss_model = transport + tf.reduce_mean(output_distance_src)
        var_ot = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='optimal_transport')
        var_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

        if __weight_decay != 0.0:  # L2
            l2_ot = __weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in var_ot])
            l2_model = __weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in var_model])
        else:
            l2_ot = l2_model = 0.0

        gradient_ot = tf.gradients(self.__loss_ot + l2_ot, var_ot)
        gradient_model = tf.gradients(self.__loss_model + l2_model, var_model)

        # optimizer
        if self.__optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.__optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        elif self.__optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        else:
            raise ValueError('unknown optimizer !!')

        # self.__train_op = optimizer.apply_gradients(zip(gradient, trainable_variables))
        with tf.control_dependencies(gradient_ot + gradient_model):
            self.__train_op_model = optimizer.apply_gradients(zip(gradient_model, var_model))
            self.__train_op_ot = optimizer.apply_gradients(zip(gradient_ot, var_ot))

        # saver
        self.__saver = tf.train.Saver()

        ##################
        # scalar summary #
        ##################
        self.__summary_src = tf.summary.merge([
            tf.summary.scalar('accuracy_src', self.__accuracy_src)
        ])

        self.__summary_tar = tf.summary.merge([
            tf.summary.scalar('accuracy_tar', self.__accuracy_tar)
        ])

        self.__summary = tf.summary.merge([
            tf.summary.scalar('meta_learning_rate', self.learning_rate),
            tf.summary.scalar('meta_keep_prob', __keep_prob),
            tf.summary.scalar('meta_weight_decay', __weight_decay),
            tf.summary.scalar('accuracy_src', self.__accuracy_src),
            tf.summary.scalar('accuracy_tar', self.__accuracy_tar),
        ])

        n_var = 0

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            sh = var.get_shape().as_list()
            self.__log('%s: %s' % (var.name, str(sh)))
            n_var += np.prod(sh)

        self.__log('total variables: %i' % n_var)

    def __log(self, statement):
        if self.__logger is not None:
            self.__logger.info(statement)

    def train(self, epoch: int):

        self.__log('checkpoint (%s), epoch (%i), learning rate (%0.7f)'
                   % (self.__checkpoint_dir, epoch, self.__ini_learning_rate))

        if self.__warm_start:
            meta = np.load('%s/meta.npz' % self.__checkpoint_dir)
            i_summary_train = int(meta['i_summary_train'])
            i_summary_valid = int(meta['i_summary_valid'])
            i_summary_valid_tar_train = int(meta['i_summary_valid_tar_train'])
            i_summary_valid_tar_valid = int(meta['i_summary_valid_tar_valid'])
            ini_epoch = int(meta['epoch'])
            learning_rate = meta['learning_rate']
        else:
            i_summary_train = 0
            i_summary_valid = 0
            i_summary_valid_tar_train = 0
            i_summary_valid_tar_valid = 0
            ini_epoch = 0
            if self.__ini_learning_rate is None:
                raise ValueError('provide learning rate !')
            learning_rate = self.__ini_learning_rate

        self.__log('checkpoint (%s), epoch (%i)' % (self.__checkpoint_dir, epoch))

        scheduler_lr = StepScheduler(
            initial_step=learning_rate,
            current_epoch=ini_epoch,
            **self.__scheduler_lr)

        for e in range(ini_epoch, ini_epoch+epoch):

            acc_train, acc_valid, acc_valid_tar_train, acc_valid_tar_valid = [], [], [], []

            #########
            # Train #
            #########
            self.__session.run([self.__iterator_ini_src], feed_dict={self.is_training: True})
            n = 0

            feed_train = {self.learning_rate: scheduler_lr(),
                          self.is_training: True}

            while True:
                n += 1
                self.__session.run([self.__iterator_ini_tar], feed_dict={self.is_training: True})
                try:

                    val = [
                        self.__accuracy_tar,
                        self.__accuracy_src,
                        self.__summary,
                        self.__train_op_ot,
                        self.__train_op_model
                    ]
                    acc_tar, acc_src, summary, _, _ = self.__session.run(val, feed_dict=feed_train)

                    acc_train.append(acc_src)
                    acc_valid_tar_train.append(acc_tar)

                    # write tensorboard writer
                    self.__writer_train.add_summary(summary, i_summary_train)
                    i_summary_train += 1  # time stamp for tf summary

                except tf.errors.OutOfRangeError:
                    if e == 0:
                        print('- %i iterations: source train' % n)
                    break

            ########
            # Test #
            ########

            # source data: valid
            self.__session.run([self.__iterator_ini_src], feed_dict={self.is_training: False})
            n = 0

            while True:
                n += 1
                try:
                    val = [
                        self.__accuracy_src,
                        self.__summary_src
                    ]
                    acc, summary = self.__session.run(val, feed_dict={self.is_training: False})
                    acc_valid.append(acc)

                    # write tensorboard writer
                    self.__writer_valid.add_summary(summary, i_summary_valid)
                    i_summary_valid += 1  # time stamp for tf summary

                except tf.errors.OutOfRangeError:
                    if e == 0:
                        print('- %i iterations: source valid' % n)
                    break

            # target data: train
            self.__session.run([self.__iterator_ini_tar], feed_dict={self.is_training: True})
            n = 0

            while True:
                n += 1
                try:
                    val = [
                        self.__accuracy_tar,
                        self.__summary_tar
                    ]
                    acc, summary = self.__session.run(val, feed_dict={self.is_training: True})
                    acc_valid_tar_train.append(acc)

                    # write tensorboard writer
                    self.__writer_valid_tar_train.add_summary(summary, i_summary_valid_tar_train)
                    i_summary_valid_tar_train += 1  # time stamp for tf summary

                except tf.errors.OutOfRangeError:
                    if e == 0:
                        print('- %i iterations: target train' % n)
                    break

            # target data: valid
            self.__session.run([self.__iterator_ini_tar], feed_dict={self.is_training: False})
            n = 0

            while True:
                n += 1
                try:
                    val = [
                        self.__accuracy_tar,
                        self.__summary_tar
                    ]
                    acc, summary = self.__session.run(val, feed_dict={self.is_training: False})
                    acc_valid_tar_valid.append(acc)

                    # write tensorboard writer
                    self.__writer_valid_tar_valid.add_summary(summary, i_summary_valid_tar_valid)
                    i_summary_valid_tar_valid += 1  # time stamp for tf summary

                except tf.errors.OutOfRangeError:
                    if e == 0:
                        print('- %i iterations: target valid' % n)
                    break

            #######
            # log #
            #######
            self.__log('epoch %i: valid (tar: %0.3f, src: %0.3f) train (tar: %0.3f, src: %0.3f)'
                       % (e,
                          float(np.mean(acc_valid_tar_valid)),
                          float(np.mean(acc_valid)),
                          float(np.mean(acc_valid_tar_train)),
                          float(np.mean(acc_train))
                          )
                       )

        self.__log('FINISH SUCCESSFULLY !!')
        self.__saver.save(self.__session, self.__checkpoint)

        np.savez('%s/meta.npz' % self.__checkpoint_dir,
                 learning_rate=learning_rate,
                 epoch=e+1,
                 i_summary_train=i_summary_train,
                 i_summary_valid=i_summary_valid,
                 i_summary_valid_tar_train=i_summary_valid_tar_train,
                 i_summary_valid_tar_valid=i_summary_valid_tar_valid
                 )








