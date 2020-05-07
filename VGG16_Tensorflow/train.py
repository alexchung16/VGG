#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/4/24 上午11:01
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from VGG16_Tensorflow.VGG16 import VGG16

from DataProcess.load_dataset import dataset_batch, get_samples

model_dir = '/home/alex/Documents/pretraining_model/vgg16'
npy_model_path = os.path.join(model_dir, 'vgg16.npy')
train_dir = '/home/alex/Documents/dataset/flower_split/train'
val_dir = '/home/alex/Documents/dataset/flower_split/val'



save_dir = os.path.join(os.getcwd(), 'model')
log_dir = os.path.join (os.getcwd(), 'logs')

input_shape = [224, 224, 3]
num_classes=5
batch_size=16
learning_rate=0.01
keep_prob = 1.0
epoch = 5

num_train_samples = get_samples(train_dir)
num_val_samples = get_samples(val_dir)

if __name__ == "__main__":

    step_per_epoch = num_train_samples // batch_size
    max_step = epoch * step_per_epoch

    vgg = VGG16(input_shape, num_classes=num_classes, learning_rate=learning_rate)

    train_image_batch, train_label_batch = \
        dataset_batch(data_dir=train_dir, batch_size=batch_size, epoch=epoch).get_next()

    val_image_batch, val_label_batch = \
        dataset_batch(data_dir=val_dir, batch_size=batch_size, epoch=epoch).get_next()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        graph = tf.get_default_graph()
        write = tf.summary.FileWriter(logdir=log_dir, graph=graph)
        saver = tf.train.Saver()

        model_variable = tf.global_variables()
        for var in model_variable:
            print(var.name, var.op.name)
            print(var.shape)
        # load weight
        # get and add histogram to summary protocol buffer
        logit_weight = graph.get_tensor_by_name(name='vgg16/fc8/Weight:0')
        tf.summary.histogram(name='logits/Weights', values=logit_weight)
        logit_biases = graph.get_tensor_by_name(name='vgg16/fc8/Bias:0')
        tf.summary.histogram(name='logits/Biases', values=logit_biases)
        # merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()

        vgg.load_weights(sess, weight_file=npy_model_path, custom_variable=['fc8'])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            if not coord.should_stop():

                # ++++++++++++++++++++++++++++++++++start training+++++++++++++++++++++++++++++++++++++++++++++++++
                # used to count the step per epoch
                step_epoch = 0
                print('Epoch: {0}/{1}'.format(0, epoch))
                for step in range(max_step):
                    train_image, train_label = sess.run([train_image_batch, train_label_batch])

                    feed_dict = vgg.fill_feed_dict(image_feed=train_image, label_feed=train_label, keep_prob=keep_prob)

                    _, train_loss, train_accuracy, summary = sess.run(
                        fetches=[vgg.train, vgg.loss, vgg.accuracy, summary_op], feed_dict=feed_dict)

                    # print training info
                    step_epoch += 1
                    print(
                        '\tstep {0}:loss value {1}  train accuracy {2}'.format(step_epoch, train_loss, train_accuracy))
                    if (step + 1) % step_per_epoch == 0:  # complete training of epoch
                        # ++++++++++++++++++++++++++++++++execute validation++++++++++++++++++++++++++++++++++++++++++++
                        # execute validation when complete every epoch
                        # validation use with all validation dataset
                        val_losses = []
                        val_accuracies = []
                        val_max_steps = int(num_val_samples / batch_size)
                        for _ in range(val_max_steps):
                            val_images, val_labels = sess.run([val_image_batch, val_label_batch])

                            feed_dict = vgg.fill_feed_dict(image_feed=val_images, label_feed=val_labels,
                                                           keep_prob=1.0)

                            val_loss, val_acc = sess.run([vgg.loss, vgg.accuracy], feed_dict=feed_dict)

                            val_losses.append(val_loss)
                            val_accuracies.append(val_acc)
                        mean_loss = np.array(val_losses, dtype=np.float32).mean()
                        mean_acc = np.array(val_accuracies, dtype=np.float32).mean()

                        print("\t{0}: epoch {1}  val Loss : {2}, val accuracy :  {3}".format(datetime.now(), epoch,
                                                                                             mean_loss, mean_acc))
                        print('Epoch: {0}/{1}'.format(step//step_per_epoch, epoch))
                        step_epoch = 0

                    write.add_summary(summary=summary, global_step=step)
                write.close()


        except Exception as e:
            print(e)
        coord.request_stop()
        coord.join(threads)
    sess.close()
    print('model training has complete')






