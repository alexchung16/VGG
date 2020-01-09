#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File visualize_conv.py
# @ Description
# @ Author alexchung
# @ Time 1/10/2019 PM 14:27


import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
from keras import optimizers, losses
from keras.preprocessing import image
from keras.applications import VGG16
from keras import backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
import cv2 as cv

# model path
model_path = os.path.join(os.getcwd(), 'model')
# train data path
data_path = os.path.join(os.getcwd(), 'data')
# image path
image_path = os.path.join(os.getcwd(), 'image')

# dataset
dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')

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


img_path = original_dataset_dir + '/cat.1700.jpg'


def visualizeAcitivitionLayer(model, layer_num, img_tensor):
    """
    visualization activation layer
    :param model:
    :param layer_num:
    :param image:
    :return:
    """
    # model instant
    layer_outputs = [layer.output for layer in model.layers[:layer_num]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activation = activation_model.predict(img_tensor)
    layer_names = []
    # layer_names = [[layer.name for layer in model.layers[i]] for  i in layer_lists]
    layer_names = [layer.name for layer in model.layers[:layer_num]]

    # the row num of image show
    n_row = 16

    for layer_name, layer_activate in zip(layer_names, activation):

        # feature num of the layer
        n_features = layer_activate.shape[-1]
        # feature image size
        length_feature_size = layer_activate.shape[1]
        width_feature_size = layer_activate.shape[2]

        n_col = n_features // n_row
        # flat image
        display_grid = np.zeros((length_feature_size*n_col, width_feature_size*n_row))

        for col in range(n_col):
            for row in range(n_row):
                channel_image = layer_activate[0, :, :, col*n_row+row]

                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                # adjustment pixel size range to between in 0 and 255
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col*width_feature_size: (col+1)*width_feature_size,
                             row*length_feature_size: (row+1)*length_feature_size] = channel_image

        # 尺度转换
        weigth_scale = 1. / length_feature_size
        height_scale = 1. / width_feature_size

        plt.figure(figsize=(weigth_scale*display_grid.shape[1], height_scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        if os.path.exists(image_path):
            pass
        else:
            os.mkdir(image_path)
        plt.savefig(image_path + '/{0}.jpg'.format(layer_name))
        # plt.show()


def generatePattern(model, layer_name, filter_index, iterate_num, img_size):
    """
    generate kernel(filter) visualize
    :param model:
    :param layer_name:
    :param filter_index:
    :param iterate_num:
    :param img_size:
    :return:
    """
    layer_output = model.get_layer(layer_name).output
    # loss function MSE
    loss = K.mean(layer_output[:, :, :, filter_index])

    # compute gradient of loss
    grads = K.gradients(loss, model.input)[0]

    # normalization grad
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # construct function to compute loss and grad
    iterate = K.function([model.input], [loss, grads])

    # contain noise range between 0 to 20
    input_img_data = np.random.random((1, img_size, img_size, 3))*20 + 128.
    # update step of gradient
    step = 1.
    for i in range(iterate_num):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocessImage(img)


def visualizeFilterPattern(model, layer_num, kernel_num,  iterate_num, img_size):
    """
    visual model kernel pattern
    :param model:
    :param layer_num: layer num
    :param kernel_num: kernel num of every layer
    :param iterate_num: iterate num
    :param img_size: input image size
    :return:
    """
    layers_name = [layer.name for layer in model.layers[1: layer_num]]
    margin = 5

    n_row = 8
    n_col = kernel_num // n_row
    for layer_name in layers_name:
        display_grid = np.zeros((n_col * img_size + 7 * margin, n_row * img_size + 7 * margin, 3))
        for i in range(n_col):
            for j in range(n_row):
                filter_img = generatePattern(model, layer_name, i*n_row+j, iterate_num, img_size)
                display_grid[i*img_size+margin*i: (i+1)*img_size+margin*i,
                            j*img_size+margin*j: (j+1)*img_size+margin*j, :] = filter_img

        # note: require the pixel intensity size between 0 and 1
        display_grid /= 255.
        plt.figure(figsize=(20, 20))
        plt.title('{0} kernel visual'.format(layer_name))

        plt.imshow(display_grid)

        if os.path.exists(image_path):
            pass
        else:
            os.mkdir(image_path)
        plt.savefig(image_path + '/{0} kernel.jpg'.format(layer_name))
        plt.show()


def deprocessImage(x):
    """
    processing image
    :param x:
    :return:
    """
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # 尺度变换到[0,1]
    return x


def visulizeClassActivateMap(model, img_tensor, cv_img):
    x = preprocess_input(img_tensor)
    predict_result = model.predict(x)
    # print('predicted: ', decode_predictions(predict_result, top=3)[0])
    # get most probability class result(top 1)
    predict_label = decode_predictions(predict_result, top=1)[0][0][1]

    max_softmax_index = np.argmax(predict_result[0])

    predict_output = model.output[:, max_softmax_index]
    last_conv_layer = model.get_layer('block5_conv3')

    grads = K.gradients(predict_output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    # sure heatmap size > 0
    heatmap = np.maximum(heatmap, 0)
    # normalize heatmap
    heatmap /= np.max(heatmap)
    # plt.imshow(heatmap)
    # plt.show()

    img = cv_img
    heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    # modify pixel size to (0, 255)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    cv.imshow('{0} origin heatmap'.format(predict_label), heatmap)
    cv.imshow('{0} mix heatmap'.format(predict_label), superimposed_img)
    cv.waitKey(0)


if __name__ == "__main__":
    # visual activation layer
    # img = image.load_img(path=img_path, target_size=(150, 150))
    # img_tensor = image.img_to_array(img)
    # img_tensor = np.expand_dims(img_tensor, axis=0)
    # img_tensor = img_tensor/255.
    # model = models.load_model(model_path+'/cnn_net.h5')

    # visualizeAcitivitionLayer(model, 6, img_tensor)
    # plt.imshow(img_tensor[0])
    # plt.show()

    # visual kernel
    # model = VGG16(weights='imagenet',
    #               include_top=False)
    # layer_name = 'block1_conv1'
    # img = generatePattern(model, layer_name, 0, 40, 64)
    # visualizeFilterPattern(model, 12, 64, 40, 64)

    # visualize heatmap of class activation
    model = VGG16(weights='imagenet')
    img_path = dataset_path+ '/commons_elephant1.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x_tensor = image.img_to_array(img)
    x = np.expand_dims(x_tensor, axis=0)
    # read image with cv.imread
    cv_img = cv.imread(img_path)
    # auto adapt modify image size
    cv_img = cv.resize(cv_img, (cv_img.shape[1]*512//cv_img.shape[0], 512))

    visulizeClassActivateMap(model, x, cv_img)

    print(cv_img.shape)



