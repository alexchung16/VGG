#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File VGG16.py
# @ Description :
# @ Author alexchung
# @ Time 11/10/2019 AM 11:20

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d, variance_scaling_initializer


class VGG16():
    """
    VGG16 model
    """
    def __init__(self, input_shape, num_classes, learning_rate):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                             name="input_images")
        # y [None,num_classes]
        self.raw_input_label = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")
        # keep_prob
        # self.keep_prob = tf.placeholder_with_default(input=1.0, shape=(), name="keep_prob")
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")


        self.global_step = tf.train.create_global_step()

        # logits
        self.logits =  self.inference(input_op=self.raw_input_data, name='vgg16')
        # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, name='loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step)
        self.accuracy = self.get_accuracy(logits=self.logits, labels=self.raw_input_label)

    def inference(self, input_op, name):
        """
        vgg16 inference
        construct static map
        :param input_op:
        :return:
        """
        #
        self.parameters = []
        with tf.variable_scope(name,reuse=None) as sc:

            self.conv1_1 = self.conv2d(input_op=input_op, scope='conv1_1', kernel_size=[3, 3], output_channels=64,
                                       padding='SAME', fineturn=False)
            self.conv1_2 = self.conv2d(input_op=self.conv1_1, scope='conv1_2', kernel_size=[3, 3], output_channels=64,
                                       padding='SAME', fineturn=False)
            self.pool1 = self.maxpool2d(input_op=self.conv1_2, scope='pool1')


            self.conv2_1 = self.conv2d(input_op=self.pool1, scope='conv2_1', kernel_size=[3, 3], output_channels=128,
                                       padding='SAME', fineturn=False)
            self.conv2_2 = self.conv2d(input_op=self.conv2_1, scope='conv2_2', kernel_size=[3, 3], output_channels=128,
                                       padding='SAME', fineturn=False)
            self.pool2 = self.maxpool2d(input_op=self.conv2_2, scope='pool2')


            self.conv3_1 = self.conv2d(input_op=self.pool2, scope='conv3_1', kernel_size=[3, 3], output_channels=256,
                                       padding='SAME', fineturn=False)
            self.conv3_2 = self.conv2d(input_op=self.conv3_1, scope='conv3_2', kernel_size=[3, 3], output_channels=256,
                                       padding='SAME', fineturn=False)
            self.conv3_3 = self.conv2d(input_op=self.conv3_2, scope='conv3_3', kernel_size=[3, 3], output_channels=256,
                                       padding='SAME', fineturn=False)
            self.pool3 = self.maxpool2d(input_op=self.conv3_3, scope='pool3')


            self.conv4_1 = self.conv2d(input_op=self.pool3, scope='conv4_1', kernel_size=[3, 3], output_channels=512,
                                       padding='SAME', fineturn=False)
            self.conv4_2 = self.conv2d(input_op=self.conv4_1, scope='conv4_2', kernel_size=[3, 3], output_channels=512,
                                       padding='SAME', fineturn=False)
            self.conv4_3 = self.conv2d(input_op=self.conv4_2, scope='conv4_3', kernel_size=[3, 3], output_channels=512,
                                       padding='SAME', fineturn=False)
            self.pool4 = self.maxpool2d(input_op=self.conv4_3, scope='pool4')


            self.conv5_1 = self.conv2d(input_op=self.pool4, scope='conv5_1', kernel_size=[3, 3], output_channels=512,
                                    padding='SAME', fineturn=False)
            self.conv5_2 = self.conv2d(input_op=self.conv5_1, scope='conv5_2', kernel_size=[3, 3], output_channels=512,
                                       padding='SAME', fineturn=False)
            self.conv5_3 = self.conv2d(input_op=self.conv5_2, scope='conv5_3', kernel_size=[3, 3], output_channels=512,
                                       padding='SAME', fineturn=False)
            self.pool5 = self.maxpool2d(input_op=self.conv5_3, scope='pool5')


            self.fc6 = self.fully_connected(input_op=self.pool5, scope='fc6', num_outputs=4096, fineturn=True)
            self.dropout1 = self.dropout(input_op=self.fc6, scope='dropout1', keep_prob=self.keep_prob)

            self.fc7 = self.fully_connected(input_op=self.dropout1, scope='fc7', num_outputs=4096, fineturn=True)
            self.dropout2 = self.dropout(input_op=self.fc7, scope='dropout2', keep_prob=self.keep_prob)
            self.fc8 = self.fully_connected(input_op=self.dropout2, scope='fc8', num_outputs=self.num_classes,
                                            fineturn=True)

            prob = tf.nn.softmax(self.fc8, name="prob")

        return prob

    def training(self, learning_rate, global_step):
        """
        train operation
        :param learnRate:
        :param globalStep:
        :param args:
        :return:
        """

        return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

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
        with tf.name_scope(name) as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='entropy')
            return tf.reduce_mean(input_tensor=cross_entropy, name='entropy_mean')

    def get_accuracy(self, logits, labels):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """
        correct_predict = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1))
        return tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))


    def fill_feed_dict(self, image_feed, label_feed, keep_prob):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed,
            self.keep_prob: keep_prob
        }
        return feed_dict


    def load_weights(self, sess, weight_file, custom_variable=None):
        """

        :param sess:
        :param weight_file:
        :param custom_variable:
        :return:
        """
        vgg16_model = np.load(file=weight_file, encoding='latin1', allow_pickle=True).item()
        vgg16_model = dict(vgg16_model)
        layers = sorted(vgg16_model.keys())

        for layer in layers:
            if layer in custom_variable:
                continue
            else:
                layer_weight, layer_bias = vgg16_model[layer]
                model_variable = tf.global_variables(scope='vgg16/{0}'.format(layer))
                # load weight and bias to graph
                sess.run([model_variable[0].assign(layer_weight), model_variable[1].assign(layer_bias)])


    def conv2d(self, input_op, scope, output_channels, kernel_size=None, strides=None, use_bias=True, padding='SAME',
                     fineturn=False, xavier=False):
        """
        convolution operation
        :param input_op:
        :param scope:
        :param filters:
        :param kernel_size:
        :param strides:
        :param use_bias:
        :param padding:
        :param parameter:
        :return:
        """
        if kernel_size is None:
            kernel_size = [3, 3]
        if strides is None:
            strides = 1
        # get feature num
        features = input_op.get_shape()[-1].value
        with tf.compat.v1.variable_scope(scope):
            weights = self.get_conv_filter(shape=[kernel_size[0], kernel_size[1], features, output_channels],
                                   trainable=fineturn, xavier=xavier)

            outputs = tf.nn.conv2d(input=input_op, filter=weights, strides=[1, strides, strides, 1], name=scope,
                                   padding=padding)
            self.parameters += [weights]

            if use_bias:
                biases = self.get_bias(shape=[output_channels], trainable=fineturn)
                outputs = tf.nn.bias_add(value=outputs, bias=biases)
                self.parameters += [biases]

            return tf.nn.relu(outputs)


    def fully_connected(self, input_op, scope, num_outputs, is_activation=True, fineturn=False, xavier=False):
        """
         full connect operation
        :param input_op:
        :param scope:
        :param num_outputs:
        :param parameter:
        :return:
        """
        # get feature num
        shape = input_op.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:
            size = shape[1]
        with tf.compat.v1.variable_scope(scope):
            flat_data = tf.reshape(tensor=input_op, shape=[-1, size], name='Flatten')

            weights =self.get_fc_weight(shape=[size, num_outputs], trainable=fineturn, xavier=xavier)
            biases = self.get_bias(shape=[num_outputs], trainable=fineturn)

            self.parameters += [weights, biases]
            if is_activation:
                 return tf.nn.relu_layer(x=flat_data, weights=weights, biases=biases)
            else:
                return tf.nn.bias_add(value=tf.matmul(flat_data, weights), bias=biases)

    def maxpool2d(self, input_op, scope, kernel_size=None, strides=None, padding='VALID'):
        """
         max pooling layer
        :param input_op:
        :param scope:
        :param kernel_size:
        :param strides_size:
        :param padding:
        :return:
        """
        with tf.compat.v1.variable_scope(scope) as scope:
            if kernel_size is None:
                ksize = [1, 2, 2, 1]
            else:
                ksize = [1, kernel_size[0], kernel_size[1], 1]
            if strides is None:
                strides = [1, 2, 2, 1]
            else:
                strides = [1, strides, strides, 1]
            return tf.nn.max_pool2d(input=input_op, ksize=ksize, strides=strides, padding=padding, name='MaxPool')


    def avgpool2d(self, input_op, scope, kernel_size=None, strides=None, padding='VALID'):
        """
        average_pool pooling layer
        :param input_op:
        :return:
        """
        with tf.compat.v1.variable_scope(scope) as scope:
            if kernel_size is None:
                ksize = [1, 2, 2, 1]
            else:
                ksize = [1, kernel_size[0], kernel_size[1], 1]
            if strides is None:
                strides = [1, 2, 2, 1]
            else:
                strides = [1, strides, strides, 1]
            return tf.nn.avg_pool2d(value=input_op, ksize=ksize, strides=strides, padding=padding, name='AvgPool')


    def dropout(self, input_op, scope, keep_prob):
        """
        dropout regularization layer
        :param inpu_op:
        :param name:
        :param keep_prob:
        :return:
        """
        with tf.compat.v1.variable_scope(scope) as sc:
            return tf.nn.dropout(input_op, keep_prob=keep_prob, name='Dropout')


    def get_conv_filter(self, shape, trainable=True, xavier=False):
        """
        convolution layer filter
        :param filter_shape:
        :return:
        """
        if xavier:
            filter = tf.get_variable(shape=shape, initializer=xavier_initializer_conv2d(),
                                    dtype=tf.float32, name='Weight',  trainable=trainable)
        else:
            filter = tf.get_variable(shape=shape, name='Weight', trainable=trainable)
        return filter


    def get_fc_weight(self, shape, trainable=True, xavier=False):
        """
        full connect layer weight
        :param weight_shape:
        :return:
        """


        if xavier:
            weight = tf.get_variable(shape=shape, initializer=xavier_initializer(), dtype=tf.float32, name='Weight',
                                     trainable=trainable)
        else:
            weight = tf.get_variable(shape=shape, trainable=trainable, name='Weight')

        return weight


    def get_bias(self, shape, trainable=True):
        """
        get bias
        :param bias_shape:

        :return:
        """
        bias = tf.get_variable(shape=shape, name='Bias', dtype=tf.float32, trainable=trainable)

        return bias

