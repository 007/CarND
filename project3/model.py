BATCH_SIZE = 64
EPOCHS = 25

# TF ordering, not TH ordering - all class docs seem to get this wrong?
INPUT_SHAPE = (160,320,3)

AUGMENT_ANGLE = 0.25 # angle offset for L/R images
OVERSTEER_ADJUSTMENT = 1.0 # magnify steering angles by this factor for training
LEARNING_RATE = 0.01
STEERING_CUTOFF = 0.01
SPEED_CUTOFF = 30

# imports
import csv
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Cropping2D, Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam

#from keras_tqdm import TQDMCallback
from keras.callbacks import ModelCheckpoint

model = None

# model
def driving_model(input_shape):
    global model
    if model == None:

        model = Sequential();
        # Crop - eliminate as much data as posible before other processing
        model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape, name='crop'))
        # downsample
        model.add(AveragePooling2D(pool_size=(2,2), name='shrink'))
        # Normalize
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='normalize'))

        # Conv2D
        model.add(Convolution2D(32, 5, 5, name='conv_5x5'))
        model.add(MaxPooling2D(name='max_pool'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='conv_activation'))

        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(256, name='fc_0'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='activation_0'))

        model.add(Dense(128, name='fc_1'))
#        model.add(BatchNormalization())
        model.add(Activation('relu', name='activation_1'))

        model.add(Dense(64, name='fc_2'))
#        model.add(BatchNormalization())
        model.add(Activation('relu', name='activation_2'))

        model.add(Dense(32, name='fc_3'))
#        model.add(BatchNormalization())
        model.add(Activation('relu', name='activation_3'))

        model.add(Dense(16, name='fc_4'))
        model.add(Activation('relu', name='activation_4'))

        model.add(Dense(8, name='fc_5'))
        model.add(Activation('relu', name='activation_5'))

        model.add(Dense(4, name='fc_6'))
        model.add(Activation('relu', name='activation_6'))

        model.add(Dense(2, name='fc_7'))
        model.add(Activation('relu', name='activation_7'))

        model.add(Dense(1, name='steering_prediction'))

        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mean_squared_error')
        model.summary()
    return model

def get_data(recording_path):
    data = []
    with open(recording_path + 'driving_log.csv') as infile:
        for row in csv.DictReader(infile):
            row['path'] = recording_path
            if abs(float(row['steering'])) > STEERING_CUTOFF and float(row['speed']) > SPEED_CUTOFF:
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
                angle = float(row['steering']) * OVERSTEER_ADJUSTMENT
                images.append(image)
                angles.append(angle)

                if augment:
                    image = np.fliplr(image)
                    images.append(image)
                    angles.append(-angle)

                    if 'left' in row and len(row['left']) > 0:
                        name = row['path'] + row['left'].strip()
                        image = get_image(name)
                        angle = float(row['steering']) * OVERSTEER_ADJUSTMENT + AUGMENT_ANGLE
                        images.append(image)
                        angles.append(angle)
                        image = np.fliplr(image)
                        images.append(image)
                        angles.append(-angle)

                    if 'right' in row and len(row['right']) > 0:
                        name = row['path'] + row['right'].strip()
                        image = get_image(name)
                        angle = float(row['steering']) * OVERSTEER_ADJUSTMENT - AUGMENT_ANGLE
                        images.append(image)
                        angles.append(angle)
                        image = np.fliplr(image)
                        images.append(image)
                        angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def train_model():
#    get_data(data_path)
#    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE, augment = True)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    train = driving_model(INPUT_SHAPE)

#    tf_session = keras.backend.get_session()
    checkpoint = ModelCheckpoint('checkpoint-{val_loss:.4f}.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    train.fit_generator(train_generator,
        samples_per_epoch=count_samples(train_samples),
        nb_val_samples=len(validation_samples),
        validation_data=validation_generator,
        nb_epoch=EPOCHS,
#        verbose=0, callbacks=[TQDMCallback()] # tqdm progress bars
        callbacks=[checkpoint]
    )
    train.save('xmodel.h5')

def pre_run():
    import tensorflow as tf
    # Creates a graph.
    with tf.device('/gpu:0'):
      a = tf.constant(1, name='a')
      b = tf.constant(1, name='b')
      c = tf.add(a, b)
    # Creates a session with allow_soft_placement
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(c)

def post_run():
    from keras import backend as K
    K.clear_session() # tensorflow/tensorflow#3388

if __name__ == '__main__':

    pre_run()

    #train_samples = get_data('/home/rmoore/src/personal/carnd/project3/recordings/data/')
    train_samples = get_data('/home/rmoore/src/personal/carnd/project3/recordings/ideal/')
    validation_samples = get_data('/home/rmoore/src/personal/carnd/project3/recordings/smooth/')
    train_model()

    post_run()
