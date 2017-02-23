#!/usr/bin/env python

AUGMENT_ANGLE = 0.25 # angle offset for L/R images
BATCH_SIZE = 64
EPOCHS = 25
INPUT_SHAPE = (160,320,3) # TF ordering, not TH ordering - all class docs seem to get this wrong?
LEARNING_RATE = 0.0001
SPEED_CUTOFF = 20
STEERING_CUTOFF = 0.01

CROP_TOP = 70
CROP_BOTTOM = 10
CROP_LEFT = 10
CROP_RIGHT = 10

# imports
import csv
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array

model = None

# model
def driving_model(input_shape):
    global model
    if model == None:

        model = Sequential();
        # Crop - eliminate as much data as posible before other processing
        model.add(Cropping2D(cropping=((CROP_TOP,CROP_BOTTOM),(CROP_LEFT,CROP_RIGHT)), input_shape=input_shape, name='crop'))
        model.add(AveragePooling2D(pool_size=(2,2), name='shrink')) # downsample

        # NVIDIA architecture
        # From https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
        # 1 normalization layer, 5 conv layers, 3 fc layers
        model.add(Lambda(lambda x: (x / 127.5) - 1, name='normalize')) # Normalize

        model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2), name='conv_5_1'))
        model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2), name='conv_5_2'))
        model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2), name='conv_5_3'))

        model.add(Convolution2D(64, 3, 3, name='conv_3_1'))
        model.add(Convolution2D(64, 3, 3, name='conv_3_2'))

        model.add(Dropout(0.4))
        model.add(Flatten(name='flatten'))

        model.add(Dense(100, activation='relu', name='fc_1'))
        model.add(Dense(50, activation='relu', name='fc_2'))
        model.add(Dense(10, activation='relu', name='fc_3'))

        model.add(Dense(1, name='steering_prediction'))

        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mean_squared_error')
        #model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()
    return model


def get_data(recording_path):
    data = []
    with open(recording_path + 'driving_log.csv') as infile:
        for row in csv.DictReader(infile):
            row['path'] = recording_path
            if abs(float(row['steering'])) >= STEERING_CUTOFF and float(row['speed']) >= SPEED_CUTOFF:
                data.append(row)
    return data


def get_image(name):
    if INPUT_SHAPE[2] == 1:
        return img_to_array(load_img(name, grayscale=True))
    else:
        return img_to_array(load_img(name))


def count_samples(samples):
    generator_count = 0
    for row in samples:
        generator_count = generator_count + 2
        if 'left' in row and len(row['left']) > 0:
            generator_count = generator_count + 2
        if 'right' in row and len(row['right']) > 0:
            generator_count = generator_count + 2
    return generator_count


def generator(samples, batch_size=32, augment = False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for row in batch_samples:

                name = row['path'] + row['center'].strip()
                image = get_image(name)
                angle = float(row['steering'])
                images.append(image)
                angles.append(angle)

                if augment:
                    image = np.fliplr(image)
                    images.append(image)
                    angles.append(-angle)

                    if 'left' in row and len(row['left']) > 0:
                        name = row['path'] + row['left'].strip()
                        image = get_image(name)
                        angle = float(row['steering']) + AUGMENT_ANGLE
                        images.append(image)
                        angles.append(angle)
                        image = np.fliplr(image)
                        images.append(image)
                        angles.append(-angle)

                    if 'right' in row and len(row['right']) > 0:
                        name = row['path'] + row['right'].strip()
                        image = get_image(name)
                        angle = float(row['steering']) - AUGMENT_ANGLE
                        images.append(image)
                        angles.append(angle)
                        image = np.fliplr(image)
                        images.append(image)
                        angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def train_model():
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE, augment = True)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    train = driving_model(INPUT_SHAPE)

    #checkpoint = ModelCheckpoint('checkpoint-{val_loss:.4f}.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    checkpoint = ModelCheckpoint('best-so-far.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    train.fit_generator(train_generator,
        samples_per_epoch=count_samples(train_samples),
        nb_val_samples=len(validation_samples),
        validation_data=validation_generator,
        nb_epoch=EPOCHS,
        callbacks=[checkpoint],
        verbose=2,
    )
    train.save('xmodel.h5')


# load CUDA and TF stuff before work starts
def pre_run():
    import tensorflow as tf
    # Creates a graph.
    with tf.device('/gpu:0'):
      a = tf.constant(0, name='a')
      x = tf.add(a, a)
    # Creates a session with allow_soft_placement
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(x)


# prevents exception on exit due to GC ordering - tensorflow/tensorflow#3388
def post_run():
    from keras import backend as K
    K.clear_session()


if __name__ == '__main__':

    pre_run()

    # train_samples = get_data('/home/rmoore/src/personal/carnd/project3/recordings/ideal/')
    # validation_samples = get_data('/home/rmoore/src/personal/carnd/project3/recordings/smooth/')
    train_samples, validation_samples = train_test_split(get_data('/home/rmoore/src/personal/carnd/project3/recordings/data/'), test_size=0.2)
    train_model()

    post_run()
