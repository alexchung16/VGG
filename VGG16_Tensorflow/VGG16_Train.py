#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File VGG16_Train.py
# @ Description :
# @ Author alexchung
# @ Time 14/10/2019 PM 20:07

import os
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
# from VGG16.VGG16 import VGG16
from VGG16.VGG16_slim import VGG16
import numpy as np
from DataProcess.read_TFRecord import reader_tfrecord, get_num_samples
from tensorflow.python.framework import graph_util


original_dataset_dir = '/home/alex/Documents/datasets/flower_photos_separate'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecord')

train_path = os.path.join(original_dataset_dir, 'train')
test_path = os.path.join(original_dataset_dir, 'test')
record_file = os.path.join(tfrecord_dir, 'image.tfrecords')
model_path = os.path.join(os.getcwd(), 'model')
model_name = os.path.join(model_path, 'vgg16.pb')
pretrain_model_dir = '/home/alex/Documents/pretraing_model/vgg16/vgg16.ckpt'
logs_dir = os.path.join(os.getcwd(), 'logs')


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('height', 224, 'Number of height size.')
flags.DEFINE_integer('width', 224, 'Number of width size.')
flags.DEFINE_integer('depth', 3, 'Number of depth size.')
flags.DEFINE_integer('num_classes', 5, 'Number of image class.')
flags.DEFINE_integer('batch_size', 128, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('epoch', 30, 'Number of epoch size.')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 1.0, 'Number of learning decay rate.')
flags.DEFINE_integer('num_epoch_per_decay', 2, 'Number epoch after each leaning rate decapy.')
flags.DEFINE_float('keep_prop', 0.8, 'Number of probability that each element is kept.')
flags.DEFINE_float('weight_decay', 0.00005, 'Number of regular scale size')
flags.DEFINE_bool('is_pretrain', True, 'if True, use pretrain model.')
flags.DEFINE_string('pretrain_model_dir', pretrain_model_dir, 'pretrain model dir.')
flags.DEFINE_string('train_dir', record_file, 'Directory to put the training data.')
flags.DEFINE_string('logs_dir', logs_dir, 'direct of summary logs.')

# pretrain model path
model_dir = '/home/alex/Documents/pretraing_model/vgg16'


def predict(model_name, image_data, input_op_name, predict_op_name):
    """
    model read and predict
    :param model_name: 
    :param image_data: 
    :param input_op_name: 
    :param predict_op_name: 
    :return: 
    """
    with tf.Graph().as_default():
        with tf.gfile.FastGFile(name=model_name, mode='rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
            _ = tf.import_graph_def(graph_def, name='')
        for index, layer in enumerate(graph_def.node):
            print(index, layer.name)

    with tf.Session() as sess:
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init_op)
        image = image_data.eval()
        input = sess.graph.get_tensor_by_name(name=input_op_name)
        output = sess.graph.get_tensor_by_name(name=predict_op_name)

        predict_softmax = sess.run(fetches=output, feed_dict={input: image})
        predict_label = np.argmax(predict_softmax, axis=1)
        return predict_label


if __name__ == "__main__":

    num_samples = get_num_samples(record_file=record_file)
    # approximate samples per epoch
    approx_sample = int((num_samples // FLAGS.batch_size) * FLAGS.batch_size)
    max_step = int((FLAGS.epoch * approx_sample) // FLAGS.batch_size)

    vgg = VGG16(input_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                num_classes=FLAGS.num_classes,
                batch_size=FLAGS.batch_size,
                learning_rate = FLAGS.learning_rate,
                decay_rate=FLAGS.decay_rate,
                num_samples_per_epoch=num_samples,
                num_epoch_per_decay=FLAGS.num_epoch_per_decay,
                keep_prob=FLAGS.keep_prop,
                weight_decay=FLAGS.weight_decay,
                is_pretrain=FLAGS.is_pretrain)

    images, labels, filenames = reader_tfrecord(record_file=FLAGS.train_dir,
                                                batch_size=FLAGS.batch_size,
                                                input_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                                                class_depth=FLAGS.num_classes,
                                                epoch=FLAGS.epoch,
                                                shuffle=True)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    # train and save model
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        # get computer graph
        graph = tf.get_default_graph()
        write = tf.summary.FileWriter(logdir=FLAGS.logs_dir, graph=graph)
        # get model variable of network
        model_variable = tf.model_variables()
        for var in model_variable:
            print(var.name)
        # get and add histogram to summary protocol buffer
        logit_weight = graph.get_tensor_by_name(name='vgg_16/fc8/weights:0')
        tf.summary.histogram(name='logits/weight', values=logit_weight)
        logit_biases = graph.get_tensor_by_name(name='vgg_16/fc8/biases:0')
        tf.summary.histogram(name='logits/biases', values=logit_biases)
        # merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()
        # load pretrain model
        if FLAGS.is_pretrain:
            # remove variable of fc8 layer from pretrain model
            custom_scope = ['vgg_16/fc8']
            for scope in custom_scope:
                variables = tf.model_variables(scope=scope)
                [model_variable.remove(var) for var in variables]
            saver = tf.train.Saver(var_list=model_variable)
            saver.restore(sess, save_path=FLAGS.pretrain_model_dir)

        # print(sess.run('vgg_16/conv1/conv1_1/biases:0'))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            if not coord.should_stop():
                # used to count the step per epoch
                step_epoch = 0
                for step in range(max_step):

                    image, label, filename = sess.run([images, labels, filenames])

                    feed_dict = vgg.fill_feed_dict(image_feed=image, label_feed=label, is_training=True)

                    _, loss_value, train_accuracy, summary = sess.run(fetches=[vgg.train, vgg.loss, vgg.accuracy, summary_op],
                                                             feed_dict=feed_dict)

                    epoch = int(step * FLAGS.batch_size / approx_sample + 1)
                    step_epoch += 1
                    if step * FLAGS.batch_size % approx_sample == 0:
                        print('Epoch: {0}/{1}'.format(epoch, FLAGS.epoch))
                        step_epoch = 0
                    print('step {0}:loss value {1}  train accuracy {2}'.format(step_epoch, loss_value, train_accuracy))

                    write.add_summary(summary=summary, global_step=step)
                write.close()

                # save model
                # get op name for save model
                input_op = vgg.raw_input_data.name
                logit_op = vgg.logits.name
                # convert variable to constant
                input_graph_def = tf.get_default_graph().as_graph_def()
                constant_graph = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                              [input_op.split(':')[0],
                                                                               logit_op.split(':')[0]])
                # save to serialize file
                with tf.gfile.FastGFile(name=model_name, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

        except Exception as e:
            print(e)
        coord.request_stop()
        coord.join(threads)
    sess.close()
    print('model training has complete')

    # # predict
    # img_path = '/home/alex/Documents/datasets/dogs_and_cat_separate/cat/dog.4.jpg'
    # image = plt.imread(img_path)
    # image = image / 255.
    # image = tf.image.resize(image, FLAGS.depth[:-1])
    # image = tf.expand_dims(image, 0)
    # predict_index = predict(model_name, image, input_op, logit_op)
    # print(predict_index)


    

    




