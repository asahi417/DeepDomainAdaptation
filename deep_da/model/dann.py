import os
import numpy as np
import tensorflow as tf
from ..util import create_log
from ..data_processing import TFRecorder
from . import base_image_fc as image_model_fc
from . import base_image_cnn as image_model_cnn
from .step_scheduler import StepScheduler
from .util_tf import variable_summaries, FlipGradientBuilder


class DANN:
    """ Domain Adversarial Neural Network """

    def __init__(self,
                 model_name: str,
                 domain_adversarial_reg: float,
                 checkpoint_dir: str,
                 path_to_tfrecord_source: str,
                 path_to_tfrecord_target: str,
                 config_feature_extractor: dict=None,
                 config_domain_classifier: dict=None,
                 config_model: dict=None,
                 learning_rate: float = None,
                 # gradient_clip: float = None,
                 batch: int = 10,
                 optimizer: str = 'sgd',
                 weight_decay: float = 0.0,
                 keep_prob: float = 1.0,
                 debug: bool = True,
                 n_thread: int = 4,
                 ckpt_epoch: int = None,
                 initializer: str='variance_scaling',
                 batch_for_test: int=1000,
                 config_scheduler_learning_rate: dict=None,
                 config_scheduler_reg: dict = None):
        """

        :param domain_adversarial_reg: regularization parameter for domain adversarial loss
        :param checkpoint_dir:
        :param path_to_tfrecord_source:
        :param path_to_tfrecord_target:
        :param learning_rate:
        :param batch:
        :param optimizer:
        :param weight_decay:
        :param keep_prob:
        :param debug:
        :param n_thread:
        :param ckpt_epoch:
        :param initializer:
        :param batch_for_test:
        """

        self.__domain_adversarial_flg = domain_adversarial_reg != 0.0
        self.__domain_adversarial_domain = domain_adversarial_reg

        self.__checkpoint_dir = checkpoint_dir
        if ckpt_epoch is None:
            self.__checkpoint = '%s/model.ckpt' % checkpoint_dir
        else:
            self.__checkpoint = '%s/model-%i.ckpt' % (checkpoint_dir, ckpt_epoch)

        # hyper parameters
        self.__ini_learning_rate = learning_rate
        # self.__clip = gradient_clip
        self.__batch = batch
        self.__optimizer = optimizer
        self.__weight_decay = weight_decay
        self.__keep_prob = keep_prob
        self.__logger = create_log('%s/log' % self.__checkpoint_dir) if debug else None

        self.__is_image = 'image' in model_name
        # tfrecorder
        self.__recorder_source = TFRecorder(path_to_tfrecord_source, debug=False, is_image=self.__is_image)
        self.__recorder_source.load_statistics()
        self.__recorder_target = TFRecorder(path_to_tfrecord_target, debug=False, is_image=self.__is_image)
        self.__recorder_target.load_statistics()

        if self.__is_image:
            # resize (data from source and target should have same sized image)
            if self.__recorder_target.image_shape[0] == self.__recorder_source.image_shape[0]:
                self.__resize = self.__recorder_target.image_shape[0]
            elif self.__recorder_target.image_shape[0] < self.__recorder_source.image_shape[0]:
                self.__resize = self.__recorder_source.image_shape[0]
            else:
                raise ValueError('source data should be larger than target.')

            # tile channel (data from source and target should have same channeled image)
            if self.__recorder_target.image_shape[2] == self.__recorder_source.image_shape[2]:
                self.__tile_channel = False
            elif self.__recorder_target.image_shape[2] == 1 and self.__recorder_source.image_shape[2] == 3:
                self.__tile_channel = True
            else:
                raise ValueError('Channel size is wired: channel of source image >= channel'
                                 'of target image and both should be 1 or 3.')

            # get models
            if model_name == 'image-cnn':
                image_model = image_model_cnn
            elif model_name == 'image-fc':
                image_model = image_model_fc
            else:
                raise ValueError('unknown model')

            self.__feature_extractor = image_model.FeatureExtractor(
                [self.__resize, self.__resize, self.__recorder_source.image_shape[2]],
                **config_feature_extractor
            )
            self.__domain_classifier = image_model.DomainClassifier(**config_domain_classifier)
            self.__model = image_model.Model(**config_model)

        else:
            raise ValueError('unknown mode')

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

        self.__n_thread = n_thread
        self.__initializer = initializer
        self.__batch_for_test = batch_for_test
        self.__scheduler_lr = config_scheduler_learning_rate
        self.__scheduler_reg = config_scheduler_reg

        self.__log('BUILD DANN GRAPH')
        self.__build_graph()

        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.__writer_train = \
            tf.summary.FileWriter('%s/summary_train' % self.__checkpoint_dir, self.__session.graph)
        self.__writer_valid = \
            tf.summary.FileWriter('%s/summary_valid' % self.__checkpoint_dir, self.__session.graph)
        self.__writer_valid_tar_train = \
            tf.summary.FileWriter('%s/summary_valid_tar_train' % self.__checkpoint_dir, self.__session.graph)
        self.__writer_valid_tar_valid = \
            tf.summary.FileWriter('%s/summary_valid_tar_valid' % self.__checkpoint_dir, self.__session.graph)

        # self.__writer_train = tf.summary.FileWriter('%s/summary_train' % self.__checkpoint_dir, self.__session.graph)
        # self.__writer_valid = tf.summary.FileWriter('%s/summary_valid' % self.__checkpoint_dir, self.__session.graph)

        # Load model
        if os.path.exists('%s.meta' % self.__checkpoint):
            self.__log('load variable from %s' % self.__checkpoint)
            self.__saver.restore(self.__session, self.__checkpoint)
            self.__warm_start = True
        else:
            os.makedirs(self.__checkpoint_dir, exist_ok=True)
            self.__session.run(tf.global_variables_initializer())
            self.__warm_start = False

    def __tfrecord(self,
                   batch,
                   is_training,
                   is_source):
        """ Get tfrecord instance

        :param batch: batch size, possibly tensor of integer
        :param is_training: boolean tensor
        :param is_source: boolean value
        :return: iterator, initializer
        """

        if is_source:
            tf_reader = self.__recorder_source.read_tf()
            path_to_tfrecord = self.__path_to_tfrecord['source']
        else:
            tf_reader = self.__recorder_target.read_tf()
            path_to_tfrecord = self.__path_to_tfrecord['target']

        tfrecord_name = tf.where(is_training, path_to_tfrecord['train'], path_to_tfrecord['valid'])
        batch = tf.where(is_training, batch, self.__batch_for_test)
        data_set_api = tf.data.TFRecordDataset(tfrecord_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(tf_reader, self.__n_thread)
        # set batch size
        buffer_size = tf.where(is_training, 10000 if is_source else 60000, 1000)
        data_set_api = data_set_api.shuffle(buffer_size=tf.cast(buffer_size, tf.int64))
        # data_set_api = data_set_api.shuffle(buffer_size=10000)
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

        # lambda for feature extractor is scheduled depend on epoch.
        # if self.__domain_adversarial_flg is False, turn off feature extractor as well
        if self.__domain_adversarial_flg:
            self.domain_adversarial_feature = tf.placeholder_with_default(0.0, [], name='domain_adversarial_reg_fe')
            domain_adversarial_feature = self.domain_adversarial_feature
            domain_adversarial_domain = self.__domain_adversarial_domain
        else:
            domain_adversarial_domain = domain_adversarial_feature = 0.0

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
        # tiling image for channel
        if self.__tile_channel:
            image_tar = tf.tile(image_tar, [1, 1, 1, 3])

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

        # domain adversarial
        with tf.variable_scope('domain_adversarial', initializer=initializer):
            flip_grad = FlipGradientBuilder()
            pred_prob_domain_src = self.__domain_classifier(
                flip_grad(feature_src, scale=domain_adversarial_feature)
            )
            pred_prob_domain_tar = self.__domain_classifier(
                flip_grad(feature_tar, scale=domain_adversarial_feature),
                reuse=True
            )

        # loss for main model
        loss_model_src = - tf.reduce_mean(tag_src * tf.log(pred_prob_src + 1e-6))
        loss_model_tar = - tf.reduce_mean(tag_tar * tf.log(pred_prob_tar + 1e-6))

        # loss for domain classification and feature extractor: target (1), source (0)
        tmp = tf.concat([tf.log(pred_prob_domain_tar + 1e-6), tf.log(1 - pred_prob_domain_src + 1e-6)], axis=0)
        loss_domain = - tf.reduce_mean(tmp)
        # loss_feature = tf.reduce_mean(tmp)

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

        ################
        # optimization #
        ################
        # # trainable variable for domain classification
        # var_domain = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='domain_adversarial')
        # # trainable variable for feature extractor
        # var_feature = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature_extraction')
        # # trainable variable for main model
        # var_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
        # var_model.extend(var_feature)

        # optimizer
        if self.__optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.__optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        elif self.__optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        else:
            raise ValueError('unknown optimizer !!')

        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        if __weight_decay != 0.0:  # L2
            l2 = __weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables])
            # l2_domain = __weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in var_domain])
            # l2_feature = __weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in var_feature])
            # l2_model = __weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in var_model])
        else:
            l2 = 0.0
            # l2_domain = l2_feature = l2_model = 0.0

        gradient = tf.gradients(loss_model_src + domain_adversarial_domain * loss_domain + l2, trainable_variables)
        self.__train_op = optimizer.apply_gradients(zip(gradient, trainable_variables))

        # gradient_model = tf.gradients(loss_model_src + l2_model, var_model)
        # gradient_domain = tf.gradients(domain_adversarial_domain * loss_domain + l2_domain, var_domain)

        # if self.__domain_adversarial_flg:
        #     gradient_feature = tf.gradients(domain_adversarial_feature * loss_feature + l2_feature, var_feature)
        #     with tf.control_dependencies(gradient_model + gradient_domain + gradient_feature):
        #         self.__train_op_model = optimizer.apply_gradients(zip(gradient_model, var_model))
        #         self.__train_op_domain = optimizer.apply_gradients(zip(gradient_domain, var_domain))
        #         self.__train_op_feature = optimizer.apply_gradients(zip(gradient_feature, var_feature))
        # else:
        #     self.__train_op_model = optimizer.apply_gradients(zip(gradient_model, var_model))
        #     self.__train_op_domain = optimizer.apply_gradients(zip(gradient_domain, var_domain))

        # saver
        self.__saver = tf.train.Saver()

        ##################
        # scalar summary #
        ##################
        self.__summary_src = tf.summary.merge([
            tf.summary.scalar('loss_model_src', loss_model_src),
            tf.summary.scalar('accuracy_src', self.__accuracy_src)
        ])

        self.__summary_tar = tf.summary.merge([
            tf.summary.scalar('loss_model_tar', loss_model_tar),
            tf.summary.scalar('accuracy_tar', self.__accuracy_tar)
        ])

        self.__summary = tf.summary.merge([
            tf.summary.scalar('meta_learning_rate', self.learning_rate),
            tf.summary.scalar('meta_domain_adversarial_reg_feature', domain_adversarial_feature),
            tf.summary.scalar('meta_domain_adversarial_reg_domain', domain_adversarial_domain),
            tf.summary.scalar('meta_keep_prob', __keep_prob),
            tf.summary.scalar('meta_weight_decay', __weight_decay),
            tf.summary.scalar('loss_model_src', loss_model_src),
            tf.summary.scalar('loss_model_tar', loss_model_tar),
            tf.summary.scalar('loss_domain', loss_domain),
            tf.summary.scalar('accuracy_src', self.__accuracy_src),
            tf.summary.scalar('accuracy_tar', self.__accuracy_tar),
            tf.summary.scalar('accuracy_domain', self.__accuracy_domain)
        ])

        n_var = 0

        for var in trainable_variables:
            sh = var.get_shape().as_list()
            self.__log('%s: %s' % (var.name, str(sh)))
            n_var += np.prod(sh)

        # self.__log('variables: feature extractor and model')
        # for var in var_model:
        #     sh = var.get_shape().as_list()
        #     self.__log('%s: %s' % (var.name, str(sh)))
        #     n_var += np.prod(sh)
        # self.__log('variables: domain adversarial')
        # for var in var_domain:
        #     sh = var.get_shape().as_list()
        #     self.__log('%s: %s' % (var.name, str(sh)))
        #     n_var += np.prod(sh)

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

        scheduler_reg = StepScheduler(
            exponential=True,
            current_epoch=ini_epoch,
            **self.__scheduler_reg)

        for e in range(ini_epoch, ini_epoch+epoch):

            acc_train, acc_valid, acc_valid_tar_train, acc_valid_tar_valid = [], [], [], []
            acc_domain = []

            #########
            # Train #
            #########
            self.__session.run([self.__iterator_ini_src], feed_dict={self.is_training: True})
            n = 0

            feed_train = {self.learning_rate: scheduler_lr(),
                          self.is_training: True,
                          self.domain_adversarial_feature: scheduler_reg()}

            while True:
                n += 1
                self.__session.run([self.__iterator_ini_tar], feed_dict={self.is_training: True})
                try:

                    if self.__domain_adversarial_flg:

                        val = [
                            self.__accuracy_tar,
                            self.__accuracy_src,
                            self.__accuracy_domain,
                            self.__summary,
                            self.__train_op
                        ]
                        acc_tar, acc_src, acc_d, summary, _ = self.__session.run(val, feed_dict=feed_train)
                    else:

                        val = [
                            self.__accuracy_tar,
                            self.__accuracy_src,
                            self.__accuracy_domain,
                            self.__summary,
                            self.__train_op
                        ]
                        acc_tar, acc_src, acc_d, summary, _ = self.__session.run(val, feed_dict=feed_train)

                    acc_train.append(acc_src)
                    acc_valid_tar_train.append(acc_tar)
                    acc_domain.append(acc_d)

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
            self.__log('epoch %i: valid (tar: %0.3f, src: %0.3f) train (tar: %0.3f, src: %0.3f, domain: % 0.3f)'
                       % (e,
                          float(np.mean(acc_valid_tar_valid)),
                          float(np.mean(acc_valid)),
                          float(np.mean(acc_valid_tar_train)),
                          float(np.mean(acc_train)),
                          float(np.mean(acc_domain))
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








