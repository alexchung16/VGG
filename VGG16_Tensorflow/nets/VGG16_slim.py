#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File VGG16_slim.py
# @ Description :
# @ Author alexchung
# @ Time 6/11/2019 AM 09:55

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


class VGG16():
    """
    VGG16 model
    """
    def __init__(self, input_shape, num_classes, batch_size, decay_rate, learning_rate, keep_prob=0.8,
                 weight_decay=0.00005, num_samples_per_epoch=None, num_epoch_per_decay=None, is_pretrain=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = int(num_samples_per_epoch / batch_size * num_epoch_per_decay)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        # self.optimizer = optimizer
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay

        self.is_pretrain = is_pretrain
        self._R_MEAN = 123.68
        self._G_MEAN = 116.78
        self._B_MEAN = 103.94
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                             name="input_images")
        self.raw_input_data = self.mean_subtraction(image=self.raw_input_data,
                                                    means=[self._R_MEAN, self._G_MEAN, self._B_MEAN])
        # y [None,num_classes]
        self.raw_input_label = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")
        self.is_training = tf.compat.v1.placeholder_with_default(input=False, shape=(), name='is_training')

        self.global_step = tf.train.create_global_step()
        # self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        # self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        # logits
        self.logits =  self.inference(inputs=self.raw_input_data, name='vgg_16')
        # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, name='loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step)
        self.accuracy = self.evaluate(logits=self.logits, labels=self.raw_input_label)

    def inference(self, inputs, name):
        """
        vgg16 inference
        construct static map
        :param input_op:
        :return:
        """
        # inputs = tf.image.per_image_standardization(inputs)
        self.parameters = []
        # inputs /= 255.
        with tf.variable_scope(name, reuse=None) as sc:
            prop = self.vgg16(inputs=inputs,
                               num_classes= self.num_classes,
                               is_training = self.is_training,
                               keep_prob = self.keep_prob,
                               scope=sc)

        return prop

    def vgg16(self, inputs,
              num_classes=None,
              is_training=True,
              keep_prob=0.5,
              reuse=None,
              scope='vgg_16'):
       with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
           # Collect outputs for conv2d, fully_connected and max_pool2d.
           with slim.arg_scope([slim.conv2d, slim.fully_connected],
                               activation_fn= tf.nn.relu,
                               weights_regularizer=slim.l2_regularizer(self.weight_decay),
                               biases_initializer=tf.zeros_initializer()):
               with tf.variable_scope('conv1', default_name='conv1'):
                   net = slim.conv2d(inputs, num_outputs=64, kernel_size=[3, 3], scope='conv1_1')
                   net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], scope='conv1_2')
               net = slim.max_pool2d(net, [2, 2], scope='pool1')
               with tf.variable_scope('conv2', default_name='conv2'):
                    net = slim.conv2d(net, num_outputs=128, kernel_size=[3, 3], scope='conv2_1')
                    net = slim.conv2d(net, num_outputs=128, kernel_size=[3, 3], scope='conv2_2')
               net = slim.max_pool2d(net, [2, 2], scope='pool2')
               with tf.variable_scope('conv3', default_name='conv3'):
                    net = slim.conv2d(net, num_outputs=256, kernel_size=[3, 3], scope='conv3_1')
                    net = slim.conv2d(net, num_outputs=256, kernel_size=[3, 3], scope='conv3_2')
                    net = slim.conv2d(net, num_outputs=256, kernel_size=[3, 3], scope='conv3_3')
               net = slim.max_pool2d(net, [2, 2], scope='pool3')
               with tf.variable_scope('conv4', default_name='conv4'):
                    net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv4_1')
                    net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv4_2')
                    net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv4_3')
               net = slim.max_pool2d(net, [2, 2], scope='pool4')
               with tf.variable_scope('conv5', default_name='conv5'):
                   net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv5_1')
                   net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv5_2')
                   net = slim.conv2d(net, num_outputs=512, kernel_size=[3, 3], scope='conv5_3')
               net = slim.max_pool2d(net, [2, 2], scope='pool5')

               # Use conv2d instead of fully_connected layers.
               net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
               net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout6')
               net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
               # dropout
               net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout7')

               net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
               logits = tf.squeeze(net, [1, 2], name='fc8/squeezed')

               # net = slim.flatten(net)
               # net = slim.fully_connected(net, num_outputs=512, scope='fc8_1')
               # net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout8')
               # logits = slim.fully_connected(inputs=net, num_outputs=num_classes, activation_fn=None, scope='fc8')
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
        # define trainable variable
        trainable_variable = None
        # trainable_scope = self.trainable_scope
        trainable_scope = ['vgg_16/conv5_1','vgg_16/conv5_2','vgg_16/conv5_3','vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8']
        # trainable_scope = []
        if self.is_pretrain and trainable_scope:
            trainable_variable = []
            for scope in trainable_scope:
                variables = tf.model_variables(scope=scope)
                [trainable_variable.append(var) for var in variables]

        learning_rate = tf.train.exponential_decay(learning_rate=learnRate, global_step=globalStep,
                                                   decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                   staircase=False)
        # according to use request of slim.batch_norm
        # update moving_mean and moving_variance when training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=globalStep,
                                                                      var_list=trainable_variable)
        return train_op


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
            loss = tf.reduce_mean(input_tensor=cross_entropy, name='loss')
            tf.summary.scalar("loss", loss)
            return loss

    def evaluate(self, logits, labels):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """
        correct_predict = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1))
        return tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))

    def fill_feed_dict(self, image_feed, label_feed, is_training):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed,
            self.is_training: is_training
        }
        return feed_dict

    def mean_subtraction(self, image, means):
        """
        subtract the means form each image channel (white image)
        :param image:
        :param mean:
        :return:
        """
        num_channels = image.get_shape()[-1]
        image = tf.cast(image, dtype=tf.float32)
        channels = tf.split(value=image, num_or_size_splits=num_channels, axis=3)
        for n in range(num_channels):
            channels[n] -= means[n]
        return tf.concat(values=channels, axis=3, name='concat_channel')


