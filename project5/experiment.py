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
    # account for stupid mpimg.imread that scales things differently
    #if np.ptp(img) <= 1.: img = (img * 255.).astype(np.uint8)
    img = (img * 255.).astype(np.uint8)
    # apply color conversion if other than 'RGB'
    if cspace == 'RGB':
        return_image = np.copy(img)
    elif cspace == 'HLS':
        return_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif cspace == 'HSV':
        return_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cspace == 'LAB':
        return_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    elif cspace == 'LUV':
        return_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif cspace == 'XYZ':
        return_image = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
    elif cspace == 'YCrCb':
        return_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif cspace == 'YUV':
        return_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    else:
        print('Unknown colorspace "{}"'.format(cspace))
        assert(False)

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
def extract_hog_features(imgs, cspace='RGB', hog_channel='ALL'):
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
    colorspace = 'YCrCb' # Can be RGB, HLS, HSV, LAB, LUV, XYZ, YCrCb, YUV
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

#        print('Using:', HOG_ORIENTATIONS, 'orientations', HOG_CELL_SIZE, 'pixels (NxN) per cell and', HOG_CELLS_PER_BLOCK, 'cells per block')
    for colorspace in [
        'HLS',
        'HSV',
        'LAB',
        'LUV',
        'RGB',
        'XYZ',
        'YCrCb',
        'YUV',
        ]:
        car_features = extract_hog_features(cars, cspace=colorspace)
        notcar_features = extract_hog_features(not_cars, cspace=colorspace)
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


        # Split up data into randomized training and test sets
        rand_state = 4 # chosen by fair dice roll, guaranteed to be random
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        # Use a linear SVC
        svc = LinearSVC()
        svc.fit(X_train, y_train)
        predict = svc.predict(X_test)

        # count false positives, negatives, exact matches
        false_positives = np.zeros_like(predict)
        false_positives[(predict == 1) & (y_test == 0)] = 1
        pos_count = np.sum(false_positives)

        false_negatives = np.zeros_like(predict)
        false_negatives[(predict == 0) & (y_test == 1)] = 1
        neg_count = np.sum(false_negatives)

        matches = np.zeros_like(predict)
        matches[predict == y_test] = 1
        match_count = np.sum(matches)

        print(colorspace,
                'accuracy is', round(svc.score(X_test, y_test) * 100, 3),
                ', false positives', pos_count,
                ', false negatives', neg_count,
                ', matches', match_count, 'out of', len(X_test)
            )

        # Save and reload to confirm it'll work IRL
        with open('svm-{}.p'.format(colorspace), 'wb') as f:
            pickle.dump(svc, f)

