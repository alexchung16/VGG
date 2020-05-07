#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : eval.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/7 下午3:24
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from PIL import Image as pil_image
# from DataProcess.vgg_preprocessing import preprocess_image

meta_path = os.path.join('../outputs/model', 'model.ckpt-2000.meta')
model_path = os.path.join('../outputs/model', 'model.ckpt-2730')

image_path = './demo/rose_0.jpg'

means =  [123.68, 116.779, 103.939]
class_name = ['sunflowers', 'roses', 'dandelion', 'daisy', 'tulips']

def image_preprocess(img_path, target_size=(224, 224), color_mode='rgb'):
    """

    :param img_path:
    :param target_size:
    :param img_type:
    :return:
    """
    img = pil_image.open(img_path)

    # convert channel
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    if color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    # resize
    width_height_tuple = (target_size[1], target_size[0])
    if img.size != width_height_tuple:
        img = img.resize(width_height_tuple, resample=pil_image.NEAREST)

    # convert tensor to array
    img_array = np.asarray(img, dtype=np.float32)

    # white process
    for channel in range(3):
        img_array[:, :, channel] -= means[channel]


    # expand dimension
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        restorer = tf.train.import_meta_graph(meta_path)

        # get graph
        graph = tf.get_default_graph()
        # restorer.restore(sess, save_path=tf.train.latest_checkpoint(checkpoint_dir='./outputs/model'))
        restorer.restore(sess, save_path=model_path)

        tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
        image_placeholder = graph.get_tensor_by_name('input_images:0')
        label_placeholder = graph.get_tensor_by_name('class_label:0')
        keep_prob_placeholder = graph.get_tensor_by_name('keep_prob:0')

        prob = graph.get_tensor_by_name('vgg16/prob:0')

        image_batch = image_preprocess(image_path, target_size=(224, 224))
        feed_dict = {image_placeholder: image_batch,
                     keep_prob_placeholder: 1.0}

        prob = sess.run(prob, feed_dict=feed_dict)
        predict_label = int(np.argmax(prob))
        print('This is a {0} with possibility {1}'.format(class_name[predict_label], prob[0][predict_label]))



