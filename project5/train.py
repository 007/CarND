#!/usr/bin/env python

import gc

import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# tweaked params from suggested 9/8/2 to 18/6/3 based on http://slideplayer.com/slide/5111258/ slide 19
HOG_ORIENTATIONS = 18
HOG_CELL_SIZE = 6
HOG_CELLS_PER_BLOCK = 3


def colorspace_convert(img, cspace):
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            return_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            return_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            return_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            return_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            return_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        return_image = np.copy(img)

    return return_image

def get_hog_features(img, orient=HOG_ORIENTATIONS, pix_per_cell=HOG_CELL_SIZE, cell_per_block=HOG_CELLS_PER_BLOCK):
    features = hog( img,
                    orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    transform_sqrt=True,
                    visualise=False,
                    feature_vector=True
                    )
    return features

def process_image_hog(img, cspace='RGB', hog_channel=0):
    # apply color conversion if other than 'RGB'
    feature_image = colorspace_convert(img, cspace)

    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel]))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel])
    # Append the new feature vector to the features list
    return hog_features

# Define a function to extract features from a list of images
def extract_hog_features(imgs, cspace='RGB', hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        features.append(process_image_hog(image, cspace, hog_channel))
    gc.collect()
    # Return list of feature vectors
    return features

if __name__ == '__main__':
    cars = glob.glob('data/vehicles/**/*.png', recursive=True)
    not_cars = glob.glob('data/non-vehicles/**/*.png', recursive=True)

    # TODO play with these values to see how your classifier
    # performs under different binning scenarios
    spatial = 32
    histbin = 32

    ### TODO: Tweak these parameters and see how the results change.
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

    time_start = time.time()
    car_features = extract_hog_features(cars, cspace=colorspace, hog_channel=hog_channel)
    notcar_features = extract_hog_features(not_cars, cspace=colorspace, hog_channel=hog_channel)
    time_end = time.time()
    print(round(time_end - time_start, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    del car_features
    del notcar_features
    gc.collect()


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', HOG_ORIENTATIONS, 'orientations', HOG_CELL_SIZE, 'pixels (NxN) per cell and', HOG_CELLS_PER_BLOCK, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    time_start = time.time()
    svc.fit(X_train, y_train)
    time_end = time.time()
    print(round(time_end - time_start, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Save this, because we can
    with open('svm.p', 'wb') as f:
        pickle.dump(svc, f)

    with open('svm.p', 'rb') as f:
        svc = pickle.load(f)


    # Check the prediction time for a single sample
    time_start = time.time()
    n_predict = 10
    print('    My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    time_end = time.time()
    print(round(time_end - time_start, 5), 'Seconds to predict', n_predict,'labels with SVC')
