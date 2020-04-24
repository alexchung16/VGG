#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : load_dataset.py
# @ Description:  https://zhuanlan.zhihu.com/p/30751039
#                 https://www.tensorflow.org/tutorials/load_data/images
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/4/24 上午11:14
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

data_dir = '/home/alex/Documents/dataset/flower_photos'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', data_dir, 'Number of height size.')


def get_file_label(data_dir, class_name=None):
    """
    get image and label
    :param dataset_path:
    :return:
    """
    image_file = []
    label_file = []
    data_dir = pathlib.Path(data_dir)
    if class_name is None:
        class_name = np.array([item.name for item in data_dir.glob('*') if item.is_dir()])

    # get file and label
    for index, name in enumerate(class_name):
        class_dir = pathlib.Path(data_dir / name)
        file_list = list(map(str, class_dir.glob('*')))
        label_list = [index] * len(file_list)
        image_file.extend(file_list)
        label_file.extend(label_list)

    # shuffle data
    # generate random index
    shuffle_index = np.random.permutation(len(image_file))
    shuffle_image = [image_file[index] for index in shuffle_index]
    shuffle_label = [label_file[index] for index in shuffle_index]

    return shuffle_image, shuffle_label


def parse_image_label(image, label, img_shape=(224, 224), label_depth=5, convert_scale=False):
    """
    parse and preprocess image label
    :param image:
    :param label:
    :param img_shape:
    :param label_depth:
    :return:
    """
    # read file to string
    image = tf.read_file(image)
    # decode image from string
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(image),
        lambda: tf.image.decode_png(image))
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    if convert_scale:
        image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image to the desired size.
    image = tf.image.resize(image, img_shape)
    # # convert dtype to tf.uint8
    # image = tf.cast(image, dtype=tf.uint8)
    label = tf.one_hot(label, depth=label_depth, on_value=1)

    return image, label


def prepare_batch(data_dir, batch_size=32, epoch=10, class_name=None, img_shape=(224, 224), label_depth=5,
                  convert_scale=True):
    """
    create dataset iterator
    :param data_dir:
    :param batch_size:
    :param epoch:
    :param class_name:
    :param img_shape:
    :param label_depth:
    :param convert_scale:
    :return:
    """
    images, labels = get_file_label(data_dir, class_name)

    filenames = tf.constant(images)
    labels = tf.constant(labels)

    # create dataset slice
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # # cache
    # dataset.cache()

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(lambda img_name, img_label: parse_image_label(image=img_name,
                                                                        label=img_label,
                                                                        img_shape=img_shape,
                                                                        label_depth=label_depth,
                                                                        convert_scale=convert_scale))

    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    dataset = dataset.shuffle(buffer_size=batch_size * 100).batch(batch_size).repeat(epoch)

    # lets the dataset fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=batch_size * 10)

    # return iterator
    return dataset.make_one_shot_iterator()


if __name__ == "__main__":

    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg'))) # 3670
    class_name = np.array([item.name for item in data_dir.glob('*') if item.is_dir()])
    print(class_name)

    list_dataset = tf.data.Dataset.list_files(str(data_dir / '*/*'))


    dataset_iterator = prepare_batch(data_dir)
    with tf.Session() as sess:

        train_image_batch, train_label_batch = dataset_iterator.get_next()
        train_image, train_label = sess.run([train_image_batch, train_label_batch])

        print(train_image.shape, train_label.shape)
        plt.imshow(train_image[0])
        plt.show()










