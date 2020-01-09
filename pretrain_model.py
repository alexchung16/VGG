#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File pretrain_model.py
# @ Description :
# @ Author alexchung
# @ Time 7/11/2019 PM 14:13

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# pretrain model path
model_dir = '/home/alex/Documents/pretraing_model/vgg16'

if __name__ == "__main__":
    vgg16_model_npy_path = os.path.join(model_dir, 'vgg16.npy')
    vgg16_model_ckpt_path = os.path.join(model_dir, 'vgg16.ckpt')

    # load .npy model
    vgg16_model = np.load(file=vgg16_model_npy_path, encoding='latin1', allow_pickle=True).item()
    vgg16_model = dict(vgg16_model)

    # get model layers
    layers = vgg16_model.keys()

    for layer, param  in vgg16_model.items():
        print(layer)
        print(param[0].shape, param[1].shape)


    with tf.Session() as sess:
        # saver.restore(sess, save_path=model_dir+'/')
        saver = tf.train.Saver()
        saver.restore(sess, save_path=vgg16_model_ckpt_path)


    """
    conv5_1
    (3, 3, 512, 512) (512,)
    fc6
    (25088, 4096) (4096,)
    conv5_3
    (3, 3, 512, 512) (512,)
    conv5_2
    (3, 3, 512, 512) (512,)
    fc8
    (4096, 1000) (1000,)
    fc7
    (4096, 4096) (4096,)
    conv4_1
    (3, 3, 256, 512) (512,)
    conv4_2
    (3, 3, 512, 512) (512,)
    conv4_3
    (3, 3, 512, 512) (512,)
    conv3_3
    (3, 3, 256, 256) (256,)
    conv3_2
    (3, 3, 256, 256) (256,)
    conv3_1
    (3, 3, 128, 256) (256,)
    conv1_1
    (3, 3, 3, 64) (64,)
    conv1_2
    (3, 3, 64, 64) (64,)
    conv2_2
    (3, 3, 128, 128) (128,)
    conv2_1
    (3, 3, 64, 128) (128,)
    """



