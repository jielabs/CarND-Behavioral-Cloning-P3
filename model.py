#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import pickle
import keras
import time
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Lambda
from keras.layers import Cropping2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import pandas as pd
import numpy as np
from PIL import Image


def read_img(img_path):
    return np.asarray(Image.open(img_path))


# Convert the image path in the log to the actual image path and read it.
def read_log_img_path(log_img_path, data_dir):
    img_name = log_img_path.split('/')[-1]
    img_path = os.path.join(data_dir, 'IMG', img_name)
    assert os.path.exists(img_path), img_path
    return read_img(img_path)


def load_csv_to_df(data_dir):
    # Load csv data to pd data frame.
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
    return log_df


# Generate the training data using all the cameras and flip images.
def training_data_generator(data_dir, dataset='train', steering_correction=0.2, batch_size = 64,
                            shuffle_data=True, add_weight=True, add_flip=True):
    log_df = load_csv_to_df(data_dir)
    total = len(log_df)
    if dataset == 'train':  # 80%
        start_idx = 0
        end_idx = int(total * 0.8)
    elif dataset == 'valid':  # 10%
        start_idx = int(total * 0.8)
        end_idx = int(total * 0.9)
    elif dataset == 'test':  # 10%
        start_idx = int(total * 0.9)
        end_idx = total
    else:
        assert False, 'Unknown dataset %s' % dataset

    while True:  # fit_generator requires infinite loop.
        # Always shuffle the data.
        log_df = load_csv_to_df(data_dir)
        if shuffle_data:
            log_df = log_df.sample(frac=1).reset_index(drop=True)
        log_df = log_df.loc[start_idx:end_idx, :]

        img_batch = []
        label_batch = []
        weight_batch = []
        for camera in ['center', 'left', 'right']:
            for idx, row in log_df.iterrows():
                img = read_log_img_path(row[camera], data_dir)
                label = row.steering
                if camera == 'left':
                    label += steering_correction
                elif camera == 'right':
                    label -= steering_correction
                if add_weight:
                    weight = 5 * (abs(label) + 0.2)
                else:
                    weight = 1
                img_batch.append(img)
                label_batch.append(label)
                weight_batch.append(weight)
                if add_flip:
                    img_batch.append(np.fliplr(img))
                    label_batch.append(-label)
                    weight_batch.append(weight)

                if len(img_batch) >= batch_size:
                    yield np.stack(img_batch), np.array(label_batch), np.array(weight_batch)
                    img_batch = []
                    label_batch = []
                    weight_batch = []


# Resize and normalize the image.
def resize_normalize(image):
    from keras.backend import tf as ktf
    resized = ktf.image.resize_images(image, (139, 139))
    resized = resized / 255.0 - 0.5
    return resized


def inception3(data_dir, model_path):
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.001
    input_size = 139
    input_shape = (160, 320, 3)
    total_samples = len(load_csv_to_df(data_dir))

    # Build the model
    car_input = Input(shape=input_shape)
    # Clip the image.
    crop_input = Cropping2D(cropping=((50, 20), (0, 0)))(car_input)
    # Resize and normalize the image.
    resized_input = Lambda(resize_normalize)(crop_input)
    # Using Inception with ImageNet pre-trained weights
    inception = InceptionV3(weights='imagenet', include_top=False,
                            input_shape=(input_size, input_size, 3))
    inp = inception(resized_input)
    out = GlobalAveragePooling2D()(inp)
    fc1 = Dense(64, activation='relu')(out)
    fc1 = Dropout(0.5)(fc1)
    predictions = Dense(1)(fc1)
    model = Model(inputs=car_input, outputs=predictions)

    # Compile the model
    model.compile(loss='mse',
                  optimizer=keras.optimizers.Nadam(lr=learning_rate),
                  metrics=['mae'])

    print(model.summary())

    # Generators for the data.
    train_generator = training_data_generator(data_dir, 'train', batch_size=batch_size)
    valid_generator = training_data_generator(data_dir, 'valid', batch_size=batch_size)
    test_generator = training_data_generator(data_dir, 'test', batch_size=batch_size)
    steps_per_epoch = int(total_samples * 0.9) / batch_size
    validation_steps = int(total_samples * 0.1) / batch_size

    # Callback functions.
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0.001, patience=6, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1)
    log_dir = os.path.expanduser('~/logs/%d' % int(time.time()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        #write_batch_performance=True,
        write_graph=True,
        write_images=False)

    # Train the model.
    history_object = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        verbose=1,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=[early_stop, checkpoint, tensorboard])

    # loss: 0.0130 - mean_absolute_error: 0.0637
    # val_loss: 0.0309 - val_mean_absolute_error: 0.0949
    # Test loss: 0.02116 - Test mae:  0.09699

    # Evaluate the model on the test set.
    score = model.evaluate_generator(test_generator, steps=validation_steps)
    print('Test loss:', score[0])
    print('Test mae:', score[1])
    pickle.dump(history_object.history, open(model_path + '.history', 'wb'))


if __name__ == '__main__':
    data_dir = '../data/data'
    model_path = 'model.h5'
    inception3(data_dir, model_path)
