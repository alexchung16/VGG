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

model_dir = '/home/alex/Documents/pretraing_model/vgg16'
vgg16_model_path = os.path.join(model_dir, 'vgg16.npy')
vgg16_model = np.load(file=vgg16_model_path, encoding='latin1', allow_pickle=True).item()
vgg16_model = dict(vgg16_model)

class VGG16():
    """
    VGG16 model
    """
    def __init__(self, input_shape, num_classes, batch_size, decay_rate, learning_rate,
                 keep_prob=0, num_samples_per_epoch=None):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = num_samples_per_epoch / batch_size * decay_rate
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        # self.optimizer = optimizer
        self.keep_prob = keep_prob
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                             name="input_images")
        # y [None,num_classes]
        self.raw_input_label = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")
        # keep_prob
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")


        self.global_step = tf.train.create_global_step()
        # self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        # self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        # logits
        self.logits =  self.inference(input_op=self.raw_input_data, name='inference')
        # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, name='loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step)
        self.evaluate_accuracy = self.evaluate_batch(logits=self.logits, labels=self.raw_input_label) / self.batch_size

    def inference(self, input_op, name):
        """
        vgg16 inference
        construct static map
        :param input_op:
        :return:
        """
        #
        self.parameters = []
        with tf.variable_scope('',reuse=None) :

            self.conv1_1 = conv2dLayer(input_op=input_op, scope='conv1_1', kernel_size=[3, 3], filters=64,
                                       padding='SAME', fineturn=True)
            self.conv1_2 = conv2dLayer(input_op=self.conv1_1, scope='conv1_2', kernel_size=[3, 3], filters=64,
                                       padding='SAME', fineturn=True)
            self.pool1 = maxPool2dLayer(input_op=self.conv1_2, scope='pool1')


            self.conv2_1 = conv2dLayer(input_op=self.pool1, scope='conv2_1', kernel_size=[3, 3], filters=128,
                                       padding='SAME', fineturn=True)
            self.conv2_2 = conv2dLayer(input_op=self.conv2_1, scope='conv2_2', kernel_size=[3, 3], filters=128,
                                       padding='SAME', fineturn=True)
            self.pool2 = maxPool2dLayer(input_op=self.conv2_2, scope='pool2')


            self.conv3_1 = conv2dLayer(input_op=self.pool2, scope='conv3_1', kernel_size=[3, 3], filters=256,
                                       padding='SAME', fineturn=True)
            self.conv3_2 = conv2dLayer(input_op=self.conv3_1, scope='conv3_2', kernel_size=[3, 3], filters=256,
                                       padding='SAME', fineturn=True)
            self.conv3_3 = conv2dLayer(input_op=self.conv3_2, scope='conv3_2', kernel_size=[3, 3], filters=256,
                                       padding='SAME', fineturn=True)
            self.pool3 = maxPool2dLayer(input_op=self.conv3_3, scope='pool3')


            self.conv4_1 = conv2dLayer(input_op=self.pool3, scope='conv4_1', kernel_size=[3, 3], filters=512,
                                       padding='SAME', fineturn=True)
            self.conv4_2 = conv2dLayer(input_op=self.conv4_1, scope='conv4_2', kernel_size=[3, 3], filters=512,
                                       padding='SAME', fineturn=True)
            self.conv4_3 = conv2dLayer(input_op=self.conv4_2, scope='conv4_3', kernel_size=[3, 3], filters=512,
                                       padding='SAME', fineturn=True)
            self.pool4 = maxPool2dLayer(input_op=self.conv4_3, scope='pool4')


            self.conv5_1 = conv2dLayer(input_op=self.pool4, scope='conv5_1', kernel_size=[3, 3], filters=512,
                                       padding='SAME', fineturn=True)
            self.conv5_2 = conv2dLayer(input_op=self.conv5_1, scope='conv5_2', kernel_size=[3, 3], filters=512,
                                       padding='SAME', fineturn=True)
            self.conv5_3 = conv2dLayer(input_op=self.conv5_2, scope='conv5_3', kernel_size=[3, 3], filters=512,
                                       padding='SAME', fineturn=True)
            self.pool5 = maxPool2dLayer(input_op=self.conv5_3, scope='pool5')

            self.flatten1 = flatten(input_op=self.pool5, scope='flatten1')

            self.fc6 = fcLayer(input_op=self.flatten1, scope='fc6', num_outputs=4096, fineturn=True)
            self.dropout1 = dropoutLayer(input_op=self.fc6, scope='dropout1', keep_prob=self.keep_prob)

            self.fc7 = fcLayer(input_op=self.dropout1, scope='fc7', num_outputs=1024, xavier=True)
            self.dropout2 = dropoutLayer(input_op=self.fc7, scope='dropout2', keep_prob=self.keep_prob)
            self.fc8 = fcLayer(input_op=self.dropout2, scope='fc8', num_outputs=self.num_classes, xavier=True)

            prop = softmaxLayer(input_op=self.fc8, scope='prop')

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
        with tf.name_scope(name) as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
            return tf.reduce_mean(input_tensor=cross_entropy, name='xentropy_mean')

    def evaluate_batch(self, logits, labels):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """
        correct_predict = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1))
        return tf.reduce_sum(tf.cast(correct_predict, dtype=tf.int32))

    # def evaluation(self, sess, feed_dict):
    #     """
    #     evaluation accuracy
    #     :param sess:
    #     :param image_pl:
    #     :param label_pl:
    #     :return:
    #     """
    #     # feed_dict = {
    #     #     self._raw_input_data : image_pl,
    #     #     self._raw_input_label : label_pl
    #     # }
    #
    #     eval_correct, loss = sess.run(fetches = self.evaluate_batch, feed_dict = feed_dict)
    #     # eval_accuracy = eval_correct / self.batch_size
    #
    #     return eval_correct

    def fill_feed_dict(self, image_feed, label_feed):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed
        }
        return feed_dict



def conv2dLayer(input_op, scope, filters, kernel_size=None, strides=None, use_bias=True, padding='SAME',
                fineturn=False, xavier=False, parameter=None):
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
        filter = getConvFilter(filter_shape=[kernel_size[0], kernel_size[1], features, filters], name=scope,
                               fineturn=fineturn, xavier=xavier)

        outputs = tf.nn.conv2d(input=input_op, filter=filter, strides=[1, strides, strides, 1], name=scope,
                               padding=padding)

        if use_bias:
            biases = getBias(bias_shape=[filters], name=scope, fineturn=fineturn, xavier=xavier)
            outputs = tf.nn.bias_add(value=outputs, bias=biases)
        #     parameter += [filter, biases]
        # else:
        #     parameter += [filters]

        return tf.nn.relu(outputs)


def fcLayer(input_op, scope, num_outputs, is_activation=True, fineturn=False, xavier=False, parameter=None):
    """
     full connect operation
    :param input_op:
    :param scope:
    :param num_outputs:
    :param parameter:
    :return:
    """
    # get feature num
    features = input_op.get_shape()[-1].value
    with tf.compat.v1.variable_scope(scope):
        weights = getFCWeight(weight_shape=[features, num_outputs], name=scope, fineturn=fineturn, xavier=xavier)
        biases = getBias(bias_shape=[num_outputs], name=scope, fineturn=fineturn, xavier=xavier)
        if is_activation:
            return tf.nn.relu_layer(x=input_op, weights=weights, biases=biases)
        else:
            return tf.nn.bias_add(value=tf.matmul(input_op, weights), bias=biases)



def maxPool2dLayer(input_op, scope, kernel_size=None, strides=None, padding='VALID'):
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


def avgPool2dLayer(input_op, scope, kernel_size=None, strides=None, padding='VALID'):
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


def flatten(input_op, scope):
    """
    flatten layer
    :param input_op:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        shape = input_op.get_shape().as_list()
        out_dim = 1
        for d in shape[1:]:
            out_dim *= d
        return tf.reshape(tensor=input_op, shape=[-1, out_dim], name='Flatten')


def dropoutLayer(input_op, scope, keep_prob):
    """
    dropout regularization layer
    :param inpu_op:
    :param name:
    :param keep_prob:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        return tf.nn.dropout(input_op, rate=1-keep_prob, name='Dropout')


def softmaxLayer(input_op, scope):
    """
    softmax layer
    :param logits:
    :param name:
    :param n_class:
    :return:
    """
    with tf.compat.v1.variable_scope(scope):
        return tf.nn.softmax(logits=input_op, name='Softmax')


def getConvFilter(filter_shape, name, fineturn=False, xavier=False):
    """
    convolution layer filter
    :param filter_shape:
    :return:
    """

    if fineturn and xavier:
        raise ValueError('fineturn and xavier no be True at the same time')
    if fineturn:
        filter = tf.constant(value=vgg16_model[name][0], name='Filter', dtype=tf.float32)
    elif xavier:
        filter = tf.get_variable(name='Filter', shape=filter_shape, initializer=xavier_initializer_conv2d(),
                                 trainable=True, dtype=tf.float32)
    else:
        filter = tf.Variable(
            initial_value=tf.random.truncated_normal(shape=filter_shape, mean=0.0, stddev=1e-1, dtype=tf.float32),
            trainable=True, name='Filter')
    return filter


def getFCWeight(weight_shape, name, fineturn=False, xavier=False):
    """
    full connect layer weight
    :param weight_shape:
    :return:
    """
    if fineturn and xavier:
        raise ValueError('fineturn and xavier no be True at the same time')
    if fineturn:
        weight = tf.constant(value=vgg16_model[name][0], name='Weight', dtype=tf.float32)
    elif xavier:
        weight = tf.get_variable(name='Weight', shape=weight_shape, initializer=xavier_initializer(),
                                 trainable=True, dtype=tf.float32)
    else:
        weight = tf.Variable(
            initial_value=tf.random.truncated_normal(shape=weight_shape, mean=0.0, stddev=1e-1, dtype=tf.float32),
            trainable=True, name='Weight')
    return weight


def getBias(bias_shape, name, fineturn=False, xavier=False):
    """
    get bias
    :param bias_shape:
    :return:
    """
    if fineturn and xavier:
        raise ValueError('fineturn and xavier no be True at the same time')
    if fineturn:
        bias = tf.constant(value=vgg16_model[name][1], name='Bias', dtype=tf.float32)
    else:
        bias = tf.Variable(initial_value=tf.constant(value=0.1, shape=bias_shape, dtype=tf.float32),
                       trainable=True, name='Bias')
    return bias

