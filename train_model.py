#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Lambda
from keras.layers import Cropping2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from sklearn.utils import shuffle
from keras.backend import tf as ktf
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

def read_img(img_path):
    return np.asarray(Image.open(img_path))

# Convert the image path in the log to the actual image path and read it.
def read_log_img_path(log_img_path, data_dir):
    img_name = log_img_path.split('/')[-1]
    img_path = os.path.join(data_dir, 'IMG', img_name)
    assert os.path.exists(img_path), img_path
    return read_img(img_path)

def load_sim_data(data_dir, camera='center', shuffle_data=True, max_limit=0):
    log_path = os.path.join(data_dir, 'driving_log.csv')
    has_header = False
    with open(log_path) as f:
        if f.read().startswith('center'):
            has_header = True
    if has_header:
        log_df = pd.read_csv(log_path)
    else:
        log_df = pd.read_csv(log_path, header=None,
                             names='center,left,right,steering,throttle,brake,speed'.split(','))
    img_list = []
    print('Load', camera, 'images')
    i = 0
    for idx, row in tqdm(log_df.iterrows()):
        img = read_log_img_path(row[camera], data_dir)
        img_list.append(img)
        i += 1
        if max_limit > 0 and i >= max_limit:
            break
    x = np.stack(img_list)
    y = np.asarray(log_df.steering)
    if max_limit > 0:
        y = y[:max_limit]
    if shuffle_data:
        x, y = shuffle(x, y)
        print('Data shuffled')
    return x, y

def get_training_data(data_folder = '../data/data'):
    x_train, y_train = load_sim_data(data_folder, 'center', shuffle_data=True, max_limit=0)
    x_test, y_test = load_sim_data(data_folder, 'left', shuffle_data=True, max_limit=640)
    return x_train, y_train, x_test, y_test

def baseline():
    batch_size = 64
    num_epochs = 4
    learning_rate = 0.001

    x_train, y_train, x_test, y_test = get_training_data()
    input_shape = x_train[0].shape  # 160, 320, 3

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Nadam(lr=learning_rate),
                  metrics=['mae', 'mse'])

    print(model.summary())

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=1,
              validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)
    model_path = 'baseline-model.h5'
    model.save(model_path)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def resize_normalize(image):
    from keras.backend import tf as ktf

    resized = ktf.image.resize_images(image, (139, 139))
    resized = resized / 255.0 - 0.5

    return resized


def inception3():
    batch_size = 64
    num_epochs = 2
    learning_rate = 0.001
    input_size = 139

    x_train, y_train, x_test, y_test = get_training_data()
    input_shape = x_train[0].shape  # 160, 320, 3

    car_input = Input(shape=input_shape)
    crop_input = Cropping2D(cropping=((50, 20), (0, 0)))(car_input)
    resized_input = Lambda(resize_normalize)(crop_input)

    # Using Inception with ImageNet pre-trained weights
    inception = InceptionV3(weights='imagenet', include_top=False,
                            input_shape=(input_size, input_size, 3))

    inp = inception(resized_input)
    out = GlobalAveragePooling2D()(inp)

    fc1 = Dense(64, activation='relu')(out)
    predictions = Dense(1)(fc1)

    model = Model(inputs=car_input, outputs=predictions)

    # Compile the model
    model.compile(loss='mse',
                  optimizer=keras.optimizers.Nadam(lr=learning_rate),
                  metrics=['mae'])

    print(model.summary())

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=1,
              validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)
    model_path = 'inception3-model.h5'
    model.save(model_path)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    inception3()