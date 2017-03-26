#!/usr/bin/env python
import cv2
import numpy as np
import pickle
import scipy.ndimage as ndimage

HEAT_THRESHOLD = 10

HISTORY_HEAT_BUFFER = None
HISTORY_BOX_BUFFER = None

def draw_heat(img, bboxes):
    draw_img = np.zeros_like(img)
    for bbox in bboxes:
        x1, y1 = bbox[0]
        x2, y2 =  bbox[1]
        draw_img[y1:y2,x1:x2] = draw_img[y1:y2,x1:x2] + 1
    # threshold to binary
    thresh = np.zeros_like(img)
    thresh[draw_img > HEAT_THRESHOLD] = 128
    return thresh

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    global HISTORY_HEAT_BUFFER
    global HISTORY_BOX_BUFFER
    if HISTORY_HEAT_BUFFER is None:
        HISTORY_HEAT_BUFFER = np.zeros_like(img)
    if HISTORY_BOX_BUFFER is None:
        HISTORY_BOX_BUFFER = np.zeros_like(img)

    # thresholded heat map
    heatmap = draw_heat(img, bboxes)
    # exponential backoff for previous frames + current frame
    HISTORY_HEAT_BUFFER = cv2.addWeighted(HISTORY_HEAT_BUFFER, 0.6, heatmap, 1, 0)

    # threshold *that* buffer to find consistent elements over several frames
    draw_img = np.zeros_like(HISTORY_HEAT_BUFFER)
    draw_img[HISTORY_HEAT_BUFFER > 160] = 255
    HISTORY_BOX_BUFFER = cv2.addWeighted(HISTORY_BOX_BUFFER, 0.9, draw_img, 1, 0)

    # now we have a buffer that should be box-able
    # generate some elements to be labeled
    labels, count = ndimage.label(draw_img)
    box_img = np.copy(img)
    if count > 0:
        # find center-of-mass for labels
        box_slices = ndimage.find_objects(labels)
        box_centers = ndimage.center_of_mass(draw_img, labels, range(1, count + 1))
        for one_slice, one_center in zip(box_slices, box_centers):
            block = labels[one_slice]
            h, w, _ = block.shape
            y, x, _ = one_center

            x1 = int(x - (w / 2.))
            x2 = int(x + (w / 2.))
            y1 = int(y - (h / 2.))
            y2 = int(y + (h / 2.))
            cv2.rectangle(box_img, (x1,y1), (x2,y2), color, thick)

    return box_img

with open('boxen.p', 'rb') as f:
    box_list = pickle.load(f)
boxy = iter(box_list)

def box_processor(img):
    block_matches = next(boxy)
    boxed = draw_boxes(img, block_matches)
    return boxed

if __name__ == '__main__':
    from moviepy.editor import VideoFileClip
    in_vid = VideoFileClip('../project4/project_video.mp4')
    out_vid = in_vid.fl_image(box_processor)
    out_vid.write_videofile('annotated_video.mp4', audio=False)
    from subprocess import call
    call(['vlc', '--play-and-exit', 'annotated_video.mp4'])

