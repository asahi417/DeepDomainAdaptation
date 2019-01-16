import os
import numpy as np
import tensorflow as tf
from ..util import create_log
from ..data_processing import TFRecorder
from . import base_image_fc as image_model_fc
from . import base_image_cnn as image_model_cnn
from .step_scheduler import StepScheduler
from .util_tf import variable_summaries


class Source:
    """ Source only model: no adaptations """

    def __init__(self,
                 model_name: str,
                 checkpoint_dir: str,
                 path_to_tfrecord_source: str,
                 path_to_tfrecord_target: str,
                 config_feature_extractor: dict = None,
                 config_model: dict = None,
                 learning_rate: float = None,
                 batch: int = 10,
                 optimizer: str = 'sgd',
                 weight_decay: float = 0.0,
                 keep_prob: float = 1.0,
                 debug: bool = True,
                 n_thread: int = 4,
                 ckpt_epoch: int = None,
                 initializer: str = 'variance_scaling',
                 batch_for_test: int = 1000,
                 config_scheduler_learning_rate: dict = None,
                 config_scheduler_reg: dict = None):
        """
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
        data_set_api = data_set_api.shuffle(buffer_size=50000)
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

        __keep_prob = tf.where(self.is_training, self.__keep_prob, 1.0)
        __weight_decay = tf.where(self.is_training, self.__weight_decay, 0.0)

        # TFRecord
        iterator_src, self.__iterator_ini_src = self.__tfrecord(self.__batch, self.is_training, is_source=True)
        iterator_tar, self.__iterator_ini_tar = self.__tfrecord(self.__batch, self.is_training, is_source=False)

        # get next input
        image_tar, tag_tar = iterator_tar.get_next()
        image_src, tag_src = iterator_src.get_next()

        ##############
        # preprocess #
        ##############
        tag_tar = tf.cast(tag_tar, tf.float32)
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

        # universal feature extraction
        with tf.variable_scope('feature_extraction', initializer=initializer):
            feature_src = self.__feature_extractor(image_src, __keep_prob)
            feature_tar = self.__feature_extractor(image_tar, reuse=True)

        # task-specific model
        with tf.variable_scope('model', initializer=initializer):
            pred_prob_src = self.__model(feature_src)
            pred_prob_tar = self.__model(feature_tar, reuse=True)

        # loss for main model
        loss_model_tar = - tf.reduce_mean(tag_tar * tf.log(pred_prob_tar + 1e-6))
        loss_model_src = - tf.reduce_mean(tag_src * tf.log(pred_prob_src + 1e-6))

        # accuracy
        self.__accuracy_tar = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(tag_tar, axis=1), tf.argmax(pred_prob_tar, axis=1)), tf.float32
            )
        )

        self.__accuracy_src = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(tag_src, axis=1), tf.argmax(pred_prob_src, axis=1)), tf.float32
            )
        )

        ################
        # optimization #
        ################
        # trainable variable for feature extractor
        var_feature = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature_extraction')
        # trainable variable for main model
        var_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
        var_model.extend(var_feature)

        # optimizer
        if self.__optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.__optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        elif self.__optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        else:
            raise ValueError('unknown optimizer !!')

        if __weight_decay != 0.0:  # L2
            l2_model = __weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in var_model])
        else:
            l2_model = 0.0

        gradient_model = tf.gradients(loss_model_src + l2_model, var_model)
        self.__train_op_model = optimizer.apply_gradients(zip(gradient_model, var_model))

        # saver
        self.__saver = tf.train.Saver()

        ##################
        # scalar summary #
        ##################
        self.__summary_src = tf.summary.merge([
            tf.summary.scalar('meta_learning_rate', self.learning_rate),
            tf.summary.scalar('meta_keep_prob', __keep_prob),
            tf.summary.scalar('meta_weight_decay', __weight_decay),
            tf.summary.scalar('accuracy_src', self.__accuracy_src),
            tf.summary.scalar('loss_model_src', loss_model_src)
        ])
        self.__summary_tar = tf.summary.merge([
            tf.summary.scalar('accuracy_tar', self.__accuracy_tar),
            tf.summary.scalar('loss_model_tar', loss_model_tar)
        ])

        n_var = 0
        self.__log('variables: feature extractor and model')
        # summary_variables = []
        for var in var_model:
            # summary_variables.append(variable_summaries(var, var.name.split(':')[0].replace('/', '-')))
            sh = var.get_shape().as_list()
            self.__log('%s: %s' % (var.name, str(sh)))
            n_var += np.prod(sh)
        # self.__summary_variable = tf.summary.merge(summary_variables)

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

        for e in range(ini_epoch, ini_epoch + epoch):

            acc_train, acc_valid, acc_valid_tar_train, acc_valid_tar_valid = [], [], [], []

            #########
            # Train #
            #########
            self.__session.run([self.__iterator_ini_src], feed_dict={self.is_training: True})
            n = 0

            while True:
                n += 1
                try:
                    val = [
                        self.__accuracy_src,
                        self.__summary_src,
                        self.__train_op_model
                    ]
                    acc, summary, _ = self.__session.run(
                        val, feed_dict={self.learning_rate: scheduler_lr(), self.is_training: True})
                    acc_train.append(acc)

                    # write tensorboard writer
                    self.__writer_train.add_summary(summary, i_summary_train)
                    i_summary_train += 1  # time stamp for tf summary

                except tf.errors.OutOfRangeError:
                    if e == 0:
                        print('- %i iterations: source train' % n)
                    break
            # summary = self.__session.run(self.__summary_variable)
            # self.__writer_train.add_summary(summary, e)  # write tensorboard writer

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
            self.__log('epoch %i: target (train: %0.3f, valid: %0.3f) source (train: %0.3f, valid: %0.3f)'
                       % (e,
                          float(np.mean(acc_valid_tar_train)),
                          float(np.mean(acc_valid_tar_valid)),
                          float(np.mean(acc_train)),
                          float(np.mean(acc_valid))
                          )
                       )

        self.__log('FINISH SUCCESSFULLY !!')
        self.__saver.save(self.__session, self.__checkpoint)

        np.savez('%s/meta.npz' % self.__checkpoint_dir,
                 learning_rate=learning_rate,
                 epoch=e + 1,
                 i_summary_train=i_summary_train,
                 i_summary_valid=i_summary_valid,
                 i_summary_valid_tar_train=i_summary_valid_tar_train,
                 i_summary_valid_tar_valid=i_summary_valid_tar_valid
                 )








