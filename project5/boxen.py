#!/usr/bin/env python
import cv2
import numpy as np
import pickle

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img

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

