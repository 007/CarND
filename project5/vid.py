#!/usr/bin/env python
import cv2
import matplotlib.image as mpimg
import numpy as np
import pickle
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

# tweaked params from suggested 9/8/2 to 18/6/3 based on http://slideplayer.com/slide/5111258/ slide 19
HOG_ORIENTATIONS = 18
HOG_CELL_SIZE = 6
HOG_CELLS_PER_BLOCK = 3
HOG_EXTRACTION_BLOCKS = (8, 8, 3, 3, 18)

# we trained on 64x64 images, so we have to crop to that size
IMAGE_BLOCK_SIZE = 64

with open('svm.p', 'rb') as f:
    svm = pickle.load(f)

def check_prediction(features):
    if svm.predict(features) == 1:
        return True
    return False

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img

with open('scaler.p', 'rb') as f:
    scaler = pickle.load(f)

def get_hog_features(img):
    features = hog( img,
                    orientations=HOG_ORIENTATIONS,
                    pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
                    cells_per_block=(HOG_CELLS_PER_BLOCK, HOG_CELLS_PER_BLOCK),
                    transform_sqrt=True,
                    visualise=False,
                    feature_vector=False
                    )
    return features


def process_image_hog(img):
    hog_features = []
    for channel in range(img.shape[2]):
        hog_features.append(get_hog_features(img[:,:,channel]))
    return hog_features
    return scaler.transform([np.ravel(hog_features)])[0]

def hog_sweep_image(img):
    # crop to 1280x384

    height, width, _ =  img.shape
    CROP_OFFSET = height - 384
    im_crop = img[CROP_OFFSET:,:,:]
    height, width, _ =  im_crop.shape

    # account for stupid mpimg.imread that scales things differently
    if np.ptp(im_crop) <= 1.: im_crop = (im_crop * 255.).astype(np.uint8)
    im_crop = cv2.cvtColor(im_crop, cv2.COLOR_RGB2YUV)

    blocks = 0
    block_matches = []
    features = process_image_hog(im_crop)

    steps_h, steps_w, _, _, _ = features[0].shape
    hog_h, hog_w, _, _, _ = HOG_EXTRACTION_BLOCKS

    for x in range(steps_w - hog_w):
        for y in range(steps_h - hog_h):
            feature_block = []
            for channel in features:
                feature_block.append(channel[y:y+hog_h,x:x+hog_w,:,:,:])
            feature_block = scaler.transform([np.ravel(feature_block)])[0]
            if check_prediction([feature_block]):
                block_matches.append([
                    (x * HOG_CELL_SIZE, (y * HOG_CELL_SIZE) + CROP_OFFSET),
                    ((x + hog_w) * HOG_CELL_SIZE, (y + hog_h) * HOG_CELL_SIZE + CROP_OFFSET)
                ])

    boxed = draw_boxes(img, block_matches)
    return boxed

if __name__ == '__main__':

    from moviepy.editor import VideoFileClip
    in_vid = VideoFileClip('project_video.mp4')
    out_vid = in_vid.fl_image(hog_sweep_image)
    out_vid.write_videofile('annotated_video.mp4', audio=False)
    from subprocess import call
    call(['vlc', '--play-and-exit', 'annotated_video.mp4'])

