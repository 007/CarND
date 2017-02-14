BATCH_SIZE = 32
EPOCHS = 10

# TF ordering, not TH ordering - all class docs seem to get this wrong?
INPUT_SHAPE = (160,320,1)

# how many outputs per input
# each row * left/right/center * +/-
GENERATOR_PERMUTATIONS = 1 * 3 * 2
AUGMENT_ANGLE = 0.2

# imports
import csv
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten, Lambda
from keras.layers.convolutional import Cropping2D, Convolution2D, UpSampling2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array

samples = []

# model
def driving_model(input_shape):
    model = Sequential();
    # Crop - eliminate as much data as posible before other processing
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape, name='crop'))
    # downsample
    model.add(AveragePooling2D(pool_size=(2,2), name='shrink'))
    # Normalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='normalize'))

    # Conv2D
    model.add(Convolution2D(32, 3, 3, name='conv_3x3'))
    model.add(MaxPooling2D(name='max_pool'))
    model.add(Activation('relu', name='activation_0'))

    model.add(Flatten())

    model.add(Dense(64, name='fc_1'))
    model.add(Activation('relu', name='activation_1'))

    model.add(Dense(32, name='fc_2'))
    model.add(Activation('softmax', name='activation_2'))

    model.add(Dense(1, name='steering_prediction'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

def get_data(recording_path):
    with open(recording_path + 'driving_log.csv') as infile:
        for row in csv.DictReader(infile):
            row['path'] = recording_path
            samples.append(row)

def get_image(name):
    if INPUT_SHAPE[2] == 1:
        return img_to_array(load_img(name, grayscale=True))
    else:
        return img_to_array(load_img(name))

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

                    name = row['path'] + row['left'].strip()
                    image = get_image(name)
                    angle = float(row['steering']) + AUGMENT_ANGLE
                    images.append(image)
                    angles.append(angle)
                    image = np.fliplr(image)
                    images.append(image)
                    angles.append(-angle)

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
    get_data('/home/ubuntu/src/personal/carnd/project3/recordings/data/')
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE, augment = True)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    train = driving_model(INPUT_SHAPE)

    train.fit_generator(train_generator, samples_per_epoch=len(train_samples) * GENERATOR_PERMUTATIONS, nb_val_samples=len(validation_samples), validation_data=validation_generator, nb_epoch=EPOCHS)
    train.save('xmodel.h5')

if __name__ == '__main__':
    train_model()
