#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/9 上午11:34
# @ Software   : PyCharm
#-------------------------------------------------------
import numpy as np
import tensorflow as tf


def predict_with_pb(model_path, image, input_op_name, logits_op_name):
    """
    model read and predict
    :param model_path:
    :param image_data:
    :param input_op_name:
    :param logits_op_name:
    :return:
    """

    with tf.Graph().as_default():
        with tf.gfile.FastGFile(name=model_path, mode='rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
            _ = tf.import_graph_def(graph_def, name='')
        for index, layer in enumerate(graph_def.node):
            print(index, layer.name)


        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            # get graph
            graph = tf.get_default_graph()
            # get tensor name
            # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
            input_op = graph.get_tensor_by_name(name=input_op_name)
            logits_op = graph.get_tensor_by_name(name=logits_op_name)

            prob = sess.run(fetches=logits_op, feed_dict={input_op: image})

            return prob