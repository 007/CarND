#!/usr/bin/env python
import matplotlib.image as mpimg
import numpy as np
import glob
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from hog import process_image_hog

# Define a function to extract features from a list of images
def extract_hog_features(imgs):
    features = []
    for file in imgs:
        image = mpimg.imread(file)
        features.append(process_image_hog(image))
    return features

if __name__ == '__main__':
    cars = glob.glob('data/vehicles/**/*.png', recursive=True)
    not_cars = glob.glob('data/non-vehicles/**/*.png', recursive=True)

    car_features = extract_hog_features(cars)
    notcar_features = extract_hog_features(not_cars)

    # Create an array stack of feature vectors
    X = np.concatenate((car_features, notcar_features), 0).astype(np.float64)

    # Define the labels vector
    y = np.concatenate((np.ones(len(car_features)), np.zeros(len(notcar_features))), axis=0)

    # Split up data into randomized training and test sets
    rand_state = 4 # chosen by fair dice roll, guaranteed to be random
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

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

    print(  'accuracy is', round(svc.score(X_test, y_test) * 100, 3),
            ', false positives', pos_count,
            ', false negatives', neg_count,
            ', matches', match_count, 'out of', len(X_test)
        )

    # Save and reload to confirm it'll work IRL
    with open('svm.p', 'wb') as f:
        pickle.dump(svc, f)

