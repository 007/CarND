#!/usr/bin/env python
import cv2
import numpy as np
import pickle

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.zeros_like(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        x1, y1 = bbox[0]
        x2, y2 =  bbox[1]
        draw_img[y1:y2,x1:x2,0] = draw_img[y1:y2,x1:x2,0] + 1 # increment red channel

    # Return the image copy with heatmap drawn
    return cv2.addWeighted(img, 0.2, draw_img * 5, 1, 0)

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

