#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File cnn.py
# @ Description
# @ Author alexchung
# @ Time 25/9/2019 AM 09:55

import os
import pickle
import shutil
import keras
from keras import layers
from keras import models
from keras import optimizers, losses, metrics
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.applications import vgg16

# model path
model_path = os.path.join(os.getcwd(), 'model')
# train data path
data_path = os.path.join(os.getcwd(), 'data')

# tensorboard log path
tb_path = os.path.join(os.getcwd(), 'logs')


# origin dataset
original_dataset_dir = '/home/alex/Documents/datasets/dogs-vs-cats/train'
# separate dataset
base_dir = '/home/alex/Documents/datasets/dogs_and_cat_separate'

# train dataset
train_dir = os.path.join(base_dir, 'train')
# validation dataset
val_dir = os.path.join(base_dir, 'validation')
# test dataset
test_dir = os.path.join(base_dir, 'test')

# train cat dataset
train_cat_dir = os.path.join(train_dir, 'cat')
# train dog dataset
train_dog_dir = os.path.join(train_dir, 'dog')

# validation cat dataset
val_cat_dir = os.path.join(val_dir, 'cat')
# validation cat dataset
val_dog_dir = os.path.join(val_dir, 'dog')

# test cat dataset
test_cat_dir = os.path.join(test_dir, 'cat')
test_dog_dir = os.path.join(test_dir, 'dog')


def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    if os.path.exists(path) is False:
        os.mkdir(path)


try:
    if os.path.exists(original_dataset_dir) is False:
        print('dataset is not exist please check the path')
    else:
        if os.path.exists(base_dir) is False:
            os.mkdir(base_dir)
            print('{0} has been created'.format(base_dir))
        else:
            print('{0} has been exist'.format(base_dir))

        makedir(train_dir)
        makedir(val_dir)
        makedir(test_dir)

        makedir(train_cat_dir)
        makedir(train_dog_dir)
        makedir(val_cat_dir)
        makedir(val_dog_dir)
        makedir(test_cat_dir)
        makedir(test_dog_dir)

except FileNotFoundError as e:
    print(e)


def structureDataset(train_num=2000, val_num=1000, test_num=1000):
    """
    structure train dataset validation dataset test dataset by separate origin dataset
    :param train_num: train dataset num
    :param val_num: validation dataset num
    :param test_num: test dataset num
    :return:
    """
    # train dataset image name list
    train_cat_list = ['cat.{0}.jpg'.format(i) for i in range(0, train_num)]
    train_dog_list = ['dog.{0}.jpg'.format(i) for i in range(0, train_num)]

    # validation dataset image name list
    val_cat_list = ['cat.{0}.jpg'.format(i) for i in range(train_num, train_num+val_num)]
    val_dog_list = ['dog.{0}.jpg'.format(i) for i in range(train_num, train_num+val_num)]

    # test dataset image name list
    test_cat_list = ['cat.{0}.jpg'.format(i) for i in range(train_num+val_num, train_num+val_num+test_num)]
    test_dog_list = ['dog.{0}.jpg'.format(i) for i in range(train_num+val_num, train_num+val_num+test_num)]

    # execute separate
    # separate train dataset
    separateDataset(train_cat_dir, train_dog_dir, train_cat_list, train_dog_list)
    # separate validation dataset
    separateDataset(val_cat_dir, val_dog_dir, val_cat_list, val_dog_list)
    # separate validation dataset
    separateDataset(test_cat_dir, test_dog_dir, test_cat_list, test_dog_list)


def separateDataset(cat_dst_dir, dog_dst_dir, cat_frame_list, dog_frame_list):
    """
    sepatate dataset
    :param cat_dst_dir:
    :param dog_dst_dir:
    :param cat_frame_list:
    :param dog_frame_list:
    :return:
    """
    for cat_frame, dog_frame in zip(cat_frame_list, dog_frame_list):
        cat_src = os.path.join(original_dataset_dir, cat_frame)
        dog_src = os.path.join(original_dataset_dir, dog_frame)
        cat_dst = os.path.join(cat_dst_dir, cat_frame)
        dog_dst = os.path.join(dog_dst_dir, dog_frame)
        shutil.copy(cat_src, cat_dst)
        shutil.copy(dog_src, dog_dst)


# image preprocessing
def imagePreprocessing():
    """
    image preprocessing
    :return:
    """
    train_data_generate = ImageDataGenerator(rescale=1./255)
    val_data_generate = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_generate.flow_from_directory(
        directory=train_dir,
        target_size=(150, 150),
        batch_size=40,
        class_mode='categorical'
    )

    val_generator = val_data_generate.flow_from_directory(
        directory=val_dir,
        target_size=(150, 150),
        batch_size=40,
        class_mode='categorical'
    )

    return train_generator, val_generator


def imageAugmentation():
    """
    image data augmentation
    Note： Only augmentation train dataset but validation dataset
    :return:
    """
    train_data_generate = ImageDataGenerator(rescale=1./255,
                                             rotation_range=40,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True,
                                             vertical_flip=True)

    # do not augmentation validation data
    val_data_generate = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_generate.flow_from_directory(
        directory=train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_data_generate.flow_from_directory(
        directory=val_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, val_generator



# def vgg16Net():
#     """
#     VGG16 Net
#     :return:
#     """
#     model = models.Sequential()
#     model.add(layers)


def cnnNet():
    """
    cnn net
    :return:
    """
    model = models.Sequential()
    # convolution layer
    # out feature map shape (148, 148, 32)
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1,
                            input_shape=(150, 150, 3)))
    # out feature map shape (74, 74, 32)
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # out feature map shape (72, 72, 64)
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # out feature map shape (36, 36, 64)
    model.add(layers.MaxPool2D((2, 2)))
    # out feature map shape (34, 34, 128)
    model.add((layers.Conv2D(128, (3, 3), activation='relu')))
    # out feature map shape (17, 17, 128)
    model.add(layers.MaxPool2D((2, 2)))
    # Dropout regularization layer
    model.add(layers.Dropout(rate=0.5))
    # FCN(Dense) layer
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def plotTrainValidationLossAccuracy(history):
    """
    Training validation loss and accuracy of epoch
    :param history: training parameter
    :return:
    """
    # history.keys() = ['val_loss', 'val_categorical_accuracy', 'loss', 'categorical_accuracy']
    history_dict = history.history
    loss = history_dict['loss']
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.plot(epochs, loss, 'bo', label='Training loss')
    ax1.plot(epochs, val_acc, 'b', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, acc, 'bo', label='Training accuracy')
    ax2.plot(epochs, val_loss, 'b', label='Validation accuracy')
    ax2.set_title('Training and validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()


def saveModel(model, model_name):
    """
    save model
    :param model: model
    :param model_name: model file name
    :return:
    """
    try:
        if os.path.exists(model_path) is False:
            os.mkdir(model_path)
            print('{0} has been created'.format(model_path))
        # save model
        model.save(os.path.join(model_path, model_name))
    except:
        raise Exception('model save failed')


def seveData(data, data_name):
    """
    save data
    :param obj: object
    :param path: save path
    :return:
    """
    try:
        if os.path.exists(data_path) is False:
            os.mkdir(data_path)
        with open(os.path.join(data_path, data_name), 'wb') as f:
            pickle.dump(data, f)
    except:
        print('data save failed')


def loadData(data_name):
    """
    load data
    :param obj: object
    :param path: save path
    :return:
    """
    try:
        if os.path.exists(data_path) is False:
            os.mkdir(data_path)
        with open(os.path.join(data_path, data_name), 'rb') as f:
            return pickle.load(f)
    except:
        print('data save failed')


if __name__ == "__main__":
    # tensorboard
    tb_cb = keras.callbacks.TensorBoard(log_dir=tb_path, histogram_freq=1, write_images=1)
    # 构建数据
    # structureDataset()
    # 获取数据信息
    train_cat_list = os.listdir(train_cat_dir)
    train_dog_list = os.listdir(train_dog_dir)
    val_cat_list = os.listdir(val_cat_dir)
    val_dog_list = os.listdir(val_dog_dir)
    test_cat_list = os.listdir(test_cat_dir)
    test_dog_list = os.listdir(test_dog_dir)
    print(len(train_cat_list), len(train_dog_list))
    print(len(val_cat_list), len(val_dog_list))
    print(len(test_cat_list), len(test_dog_list))

    # show sample image
    # img = mpimg.imread(os.path.join(train_cat_dir, 'cat.0.jpg'))
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()

    # model
    model = cnnNet()
    print(model.summary())
    weight, bias = model.get_layer(name='conv2d_1').get_weights()
    print(weight.shape)
    print(weight)
    # print(model.summary())
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    train_generator, val_generator = imagePreprocessing()
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=val_generator,
        validation_steps=50,
    )
    # # save model
    # saveModel(model, 'cnn_net.h5')
    # # save train history index
    # seveData(history.history, 'cnn_net.pkl')
    # hist = loadData('history.pkl')
    # plotTrainValidationLossAccuracy((hist))



