#!/usr/bin/env python

import cv2
import glob
import json
import numpy as np

from local_debug_helper import imgprint, imgprint_h

CHESS_X = 9 # inner-horizontal corners on calibration images
CHESS_Y = 6 # inner-vertical corners on calibration images

def save_camera_calibration(mtx, dist):
    # save to camera_calibration.json
    json_data = {'mtx': mtx.tolist(), 'dist': dist.tolist()}
    with open('camera_calibration.json', 'w') as f:
        json.dump(json_data, f, indent=2, sort_keys=True)

def load_camera_calibration():
    # load from camera_calibration.json
    # TODO: cache this after first load?
    with open('camera_calibration.json', 'r') as f:
        json_data = json.load(f)
    # reconstitute as np arrays vs lists
    mtx = np.array(json_data['mtx'])
    dist = np.array(json_data['dist'])
    return mtx, dist

""" Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. """
def compute_calibration(file_array):
    calibration_shape = (1280,720) # HARD CODED
    obj_points = []
    img_points = []

    objp = np.zeros((CHESS_Y*CHESS_X, 3), np.float32)
    objp [:,:2] = np.mgrid[0:CHESS_X,0:CHESS_Y].T.reshape(-1,2)
    for fname in file_array:
        print("processing {}".format(fname))
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (CHESS_X, CHESS_Y), None)
        if ret == True:
            img_points.append(corners)
            obj_points.append(objp)
    _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, calibration_shape, None, None)

    save_camera_calibration(mtx, dist)


""" Apply a distortion correction to raw images. """
def correct_distortion(img):
    mtx, dist = load_camera_calibration()
    return cv2.undistort(img, mtx, dist, None, mtx)

""" Use color transforms, gradients, etc., to create a thresholded binary image. """
def image_to_threshold(img,  thresh_min=100, thresh_max=999):

    foo = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bar = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # 1) Convert to grayscale
    # diff saturation channels
    gray = foo[:,:,1] - bar[:,:,2]
    # 2) Take the derivative in x
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude is between thresh_min and _max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


""" Apply a perspective transform to rectify binary image ("birds-eye view"). """
def perspective_warp_lane(img):
    w,h = 1280, 720
    border = 128

    from_shape = np.float32([
        [ 595,  450],
        [ 690,  450],
        [1035,  675],
        [ 275,  675],
        ])
    to_shape = np.float32([
        [ border, border ],
        [w - border, border],
        [w - border, h - border],
        [border, h - border],
        ])
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(from_shape, to_shape)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

""" Detect lane pixels and fit to find the lane boundary. """
""" Determine the curvature of the lane and vehicle position with respect to center. """
""" Warp the detected lane boundaries back onto the original image. """
""" Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position. """


def pipeline_init():
    # skip for now
    # compute_calibration(glob.glob('./camera_cal/calibration*.jpg'))
    img = cv2.imread('./camera_cal/calibration2.jpg')
    dst = correct_distortion(img)
    cv2.imwrite('./output/calibration_input.jpg', img)
    cv2.imwrite('./output/calibration_output.jpg', dst)

def pipeline(input_image):
    traffic_image = cv2.imread(input_image)

    corrected = correct_distortion(traffic_image)
    cv2.imwrite('./output/traffic_calibrated.jpg', corrected)
    imgprint(corrected)

    threshold = image_to_threshold(corrected)
    cv2.imwrite('./output/traffic_tresholded.jpg', threshold)
    #imgprint(threshold)

    warped = perspective_warp_lane(threshold)
    cv2.imwrite('./output/traffic_perspective.jpg', warped)
    imgprint(warped)


if __name__ == '__main__':
    pipeline_init()
    pipeline('test_images/straight_lines2.jpg')

