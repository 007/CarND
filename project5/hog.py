import numpy as np
import pickle
import cv2
from skimage.feature import hog

# tweaked params from suggested 9/8/2 to 18/6/3 based on http://slideplayer.com/slide/5111258/ slide 19
HOG_ORIENTATIONS = 18
HOG_CELL_SIZE = 6
HOG_CELLS_PER_BLOCK = 3

with open('scaler.p', 'rb') as f:
    scaler = pickle.load(f)

def get_hog_features(img):
    features = hog( img,
                    orientations=HOG_ORIENTATIONS,
                    pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
                    cells_per_block=(HOG_CELLS_PER_BLOCK, HOG_CELLS_PER_BLOCK),
                    transform_sqrt=True,
                    visualise=False,
                    feature_vector=True
                    )
    return features

def process_image_hog(img):
    # account for stupid mpimg.imread that scales things differently
    if np.ptp(img) <= 1.: img = (img * 255.).astype(np.uint8)
    feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_features.append(get_hog_features(feature_image[:,:,channel]))
    return scaler.transform([np.ravel(hog_features)])[0]

