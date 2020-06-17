#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : load_graph.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/6/1 下午2:25
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import tensorflow  as tf


model_dir = '../outputs/model'

meta_graph = os.path.join(model_dir, 'model.ckpt-2000.meta')

if __name__ == "__main__":

    with tf.Session() as sess:
        restore = tf.train.import_meta_graph(meta_graph)
        restore.restore(sess, save_path=tf.train.latest_checkpoint(model_dir))

        for var in tf.global_variables():
            print(var.op.name)
            print(var.eval().shape)