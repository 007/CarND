#!/usr/bin/env python

# Things to play with
AUGMENT_ANGLE = 0.3 # angle offset for L/R images
BATCH_SIZE = 32
EPOCHS = 25
INPUT_SHAPE = (160,320,3) # TF ordering, not TH ordering - most class docs seem to get this wrong?
LEARNING_RATE = 0.01 # higher learning rate recommended due to dropout

# how much to crop off each edge
CROP_TOP = 80
CROP_BOTTOM = 10
CROP_LEFT = 0
CROP_RIGHT = 0


# imports
import csv
import numpy as np
import sklearn # only need for one function

from keras.callbacks import ModelCheckpoint
from keras.engine.topology import InputLayer
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array, flip_axis


# FC layer helper
# wish BN was an inline param like Dense(size, activation='elu', batch_norm=True)
def fc_helper(model, size):
    model.add(Dense(size))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.2))

# model
def driving_model(input_shape):
    model = Sequential();

    # explicit input layer makes summary easier to understand
    model.add(InputLayer(input_shape=input_shape, name='start'))

    # crop - eliminate as much data as posible before other processing
    crop_shape = ((CROP_TOP,CROP_BOTTOM),(CROP_LEFT,CROP_RIGHT))
    model.add(Cropping2D(cropping=crop_shape, name='crop'))

    # very much NVIDIA architecture
    # From https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    # 1 normalization layer, 5 conv layers, 3 fc layers
    # added dropout and batch normalization, and used elu vs relu/tanh/sig

    model.add(Lambda(lambda x: (x / 127.5) - 1, name='normalize'))
    model.add(Dropout(0.2, name='dropout_0'))

    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2), name='conv_5_1'))
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2), name='conv_5_2'))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2), name='conv_5_3'))

    model.add(Convolution2D(64, 3, 3, name='conv_3_1'))
    model.add(Convolution2D(64, 3, 3, name='conv_3_2'))

    model.add(Flatten(name='flatten'))

    fc_helper(model, 100)
    fc_helper(model, 50)
    fc_helper(model, 10)

    model.add(Dense(1, name='steering_prediction'))

    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mean_squared_error')
    model.summary() # this should be mandatory for all submissions!

    return model

# load CSV
def get_data(recording_path):
    columns = ('center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed')
    data = []
    with open(recording_path + 'driving_log.csv') as infile:
        for row in csv.DictReader(infile, fieldnames=columns):
            data.append(row)
    return data

# load image
def get_image(name):
    return img_to_array(load_img(name))

# count samples that will be returned after augmentation
def count_samples(samples, augment=False):
    generator_count = 0
    for row in samples:
        generator_count = generator_count + 2 # always do center + flip
        if augment:
            if 'left' in row and len(row['left']) > 0:
                generator_count = generator_count + 2
            if 'right' in row and len(row['right']) > 0:
                generator_count = generator_count + 2
    return generator_count

# load image, adjust steering angle, flip image and angle
def augmentation_helper(row, index, offset_multiplier=1.0):
    images, angles = [], []
    if index in row and len(row[index]) > 0:
        name = row[index].strip()
        image = get_image(name)
        angle = float(row['steering']) + (AUGMENT_ANGLE * offset_multiplier)
        images.append(image)
        angles.append(angle)
        image = flip_axis(image, 1)
        images.append(image)
        angles.append(angle * -1.0)
    return (images, angles)

# generative wrapper based on class material
def generator(samples, batch_size=32, augment=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for row in batch_samples:
                img, ang = augmentation_helper(row, 'center', 0)
                images.extend(img)
                angles.extend(ang)

                if augment:
                    img, ang = augmentation_helper(row, 'left', 1)
                    images.extend(img)
                    angles.extend(ang)

                    img, ang = augmentation_helper(row, 'right', -1.0)
                    images.extend(img)
                    angles.extend(ang)

            X_train = np.array(images)
            y_train = np.array(angles)
            # These are actuall batch_size * augmentation count, not ideal
            yield sklearn.utils.shuffle(X_train, y_train)


def train_model():
    train_count = count_samples(train_samples, augment=True)
    valid_count = count_samples(validation_samples)

    print('-=' * 40)
    print('  Training samples: {}'.format(len(train_samples)))
    print(' With augmentation: {}'.format(train_count))
    print('Validation samples: {}'.format(valid_count))
    print('-=' * 40)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE, augment=True)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    train = driving_model(INPUT_SHAPE)

    checkpoint = ModelCheckpoint('best-so-far.h5',
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
    train.fit_generator(train_generator,
        samples_per_epoch=train_count,
        nb_val_samples=valid_count,
        validation_data=validation_generator,
        nb_epoch=EPOCHS,
        callbacks=[checkpoint],
        verbose=2,
    )
    train.save('xmodel.h5')

# load CUDA and TF stuff before work starts
# gives cleaner output for stats / summary / training
def pre_run():
    import tensorflow as tf # boo to function-level imports
    with tf.device('/gpu:0'):
      a = tf.constant(0, name='a')
      x = tf.add(a, a)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(x)

# prevents exception on exit due to GC ordering - tensorflow/tensorflow#3388
def post_run():
    from keras import backend as K
    K.clear_session()

# Do stuff
if __name__ == '__main__':

    pre_run()

    train_samples = []
    train_samples.extend(get_data('/home/rmoore/src/personal/carnd/project3/recordings/tons/'))
    train_samples.extend(get_data('/home/rmoore/src/personal/carnd/project3/recordings/recovery/'))
    validation_samples = get_data('/home/rmoore/src/personal/carnd/project3/recordings/data/')
    train_model()

    post_run()
