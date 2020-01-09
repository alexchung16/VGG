#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File VGG16_Pretrain.py
# @ Description :
# @ Author alexchung
# @ Time 11/11/2019 AM 09：33


import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


class VGG16():
    """
    VGG16 pretrain model
    """
    def __init__(self, input_shape, num_classes, batch_size, decay_rate, learning_rate,
                 keep_prob=0.5, num_samples_per_epoch=None, num_epoch_per_decay=None, is_pretrain=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = int(num_samples_per_epoch / batch_size * num_epoch_per_decay)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        # self.optimizer = optimizer
        self.keep_prob = keep_prob
        self.is_pretrain = is_pretrain
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                             name="input_images")
        # y [None,num_classes]
        self.raw_input_label = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")
        self.is_training = tf.compat.v1.placeholder_with_default(input=False, shape=(), name='is_training')

        self.global_step = tf.train.create_global_step()
        # self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        # self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        # logits
        self.logits =  self.inference(inputs=self.raw_input_data, name='inference')
        # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, name='loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step)
        self.evaluate_accuracy = self.evaluate_batch(logits=self.logits, labels=self.raw_input_label) / self.batch_size


    def inference(self, inputs, name):
        """
        vgg16 inference
        construct static map
        :param input_op:
        :return:
        """
        #
        self.parameters = []

        with tf.variable_scope('',reuse=None) :
            prop = self.vgg_16(inputs=inputs,
                               num_classes= self.num_classes,
                               is_training = self.is_training,
                               keep_prob = self.keep_prob,
                               is_pretrain=self.is_pretrain)

        return prop

    def vgg_16(self, inputs,
              num_classes=None,
              is_training=True,
              keep_prob=0.5,
              reuse=None,
              scope='vgg_16',
              fc_conv_padding='VALID',
              is_pretrain=False):

       with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
           # Collect outputs for conv2d, fully_connected and max_pool2d.
           with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
               if is_pretrain:
                   net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', trainable=False)
                   net = slim.max_pool2d(net, [2, 2], scope='pool1')
                   net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', trainable=False)
                   net = slim.max_pool2d(net, [2, 2], scope='pool2')
                   net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', trainable=False)
                   net = slim.max_pool2d(net, [2, 2], scope='pool3')
                   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4', trainable=False)
                   net = slim.max_pool2d(net, [2, 2], scope='pool4')
                   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5', trainable=False)
                   net = slim.max_pool2d(net, [2, 2], scope='pool5')

                   # Use conv2d instead of fully_connected layers.
                   net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6', trainable=False)
                   net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout6')
                   net = slim.conv2d(net, 4096, [1, 1], scope='fc7', trainable=False)
                   # dropout
                   net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout7')
               else:
                   net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                   net = slim.max_pool2d(net, [2, 2], scope='pool1')
                   net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                   net = slim.max_pool2d(net, [2, 2], scope='pool2')
                   net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                   net = slim.max_pool2d(net, [2, 2], scope='pool3')
                   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                   net = slim.max_pool2d(net, [2, 2], scope='pool4')
                   net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                   net = slim.max_pool2d(net, [2, 2], scope='pool5')

                   # Use conv2d instead of fully_connected layers.
                   net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
                   net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout6')
                   net = slim.conv2d(net, 4096, [1, 1], scope='fc7',)
                   # dropout
                   net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout7')

               # net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
               # logits = tf.squeeze(net, [1, 2], name='fc8/squeezed')
               net = slim.flatten(net)
               net = slim.fully_connected(net, num_outputs=512, scope='fc8_1')
               net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout8')
               logits = slim.fully_connected(inputs=net, num_outputs=num_classes, activation_fn=None, scope='fc8_2')
               # softmax
               prop = slim.softmax(logits=logits, scope='softmax')
               return prop



    def training(self, learnRate, globalStep):
        """
        train operation
        :param learnRate:
        :param globalStep:
        :param args:
        :return:
        """
        learning_rate = tf.train.exponential_decay(learning_rate=learnRate, global_step=globalStep,
                                                   decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                   staircase= False)
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=globalStep)

    # def predict(self):
    #     """
    #     predict operation
    #     :return:
    #     """
    #
    #     return tf.cast(self.logits, dtype=tf.float32, name="predicts")


    def losses(self, logits, labels, name):
        """
        loss function
        :param logits:
        :param labels:
        :return:
        """
        with tf.name_scope(name):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='entropy')
            return tf.reduce_mean(input_tensor=cross_entropy, name='loss')

    def evaluate_batch(self, logits, labels):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """
        correct_predict = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1))
        return tf.reduce_sum(tf.cast(correct_predict, dtype=tf.int32))

    def fill_feed_dict(self, image_feed, label_feed, is_training):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed,
            self.is_training: is_training
        }
        return feed_dict


if __name__ == "__main__":
    num_classes = 1000
    is_training = True
    dropout_keep_prob = 0.5
    spatial_squeeze = True
    reuse = None
    scope = 'vgg_16'
    fc_conv_padding = 'VALID'
    global_pool = False

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='inputs')

    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            # net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            with tf.variable_scope('conv1', default_name='conv1'):
                net = slim.conv2d(inputs, num_outputs=64, kernel_size=[3, 3], scope='conv1_1')
                net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            # net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            with tf.variable_scope('conv2', default_name='conv2'):
                net = slim.conv2d(net, num_outputs=128, kernel_size=[3, 3], scope='conv2_1')
                net = slim.conv2d(net, num_outputs=128, kernel_size=[3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            with tf.variable_scope('conv3', default_name='conv3'):
                net = slim.conv2d(net, num_outputs=256, kernel_size=[3, 3], scope='conv3_1')
                net = slim.conv2d(net, num_outputs=256, kernel_size=[3, 3], scope='conv3_2')
                net = slim.conv2d(net, num_outputs=256, kernel_size=[3, 3], scope='conv3_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            with tf.variable_scope('conv4', default_name='conv4'):
                net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv4_1')
                net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv4_2')
                net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv4_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            with tf.variable_scope('conv5', default_name='conv5'):
                net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv5_1')
                net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv5_2')
                net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv5_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

            # dropout
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
            logits = tf.squeeze(net, [1, 2], name='fc8/squeezed')

            # softmax
            logit = slim.softmax(logits=logits, scope='softmax')

    with tf.Session() as sess:
        # inputs = tf.random_uniform(shape=(6, 224, 224, 3))

        sess.run(tf.global_variables_initializer())

        graph = tf.get_default_graph()
        # get model variable
        model_variable = tf.model_variables()
        for var in model_variable:
            print(var.name, var.shape)

        print(sess.run('vgg_16/conv1/conv1_1/biases:0'))
        print(sess.run('vgg_16/conv1/conv1_1/weights:0').shape)

        # remove variables of the custom layer from raw variables
        custom_scope = ['vgg_16/fc8']
        for scope in custom_scope:
            variables = tf.model_variables(scope=scope)
            [model_variable.remove(var) for var in variables]

        saver = tf.train.Saver(var_list=model_variable)
        saver.restore(sess, save_path='/home/alex/Documents/pretraing_model/vgg16/vgg16.ckpt')

        print(sess.run('vgg_16/conv1/conv1_1/biases:0'))

