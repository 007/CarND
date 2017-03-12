#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import json

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
    mtx2, dist2 = load_camera_calibration()

    for fname in file_array:
        print("processing {}".format(fname))
        img = cv2.imread(fname)
        dst = cv2.undistort(img, mtx2, dist2, None, mtx2)
        imgprint_h((img, dst))


compute_calibration(glob.glob('./camera_cal/calibration*.jpg'))




""" Apply a distortion correction to raw images. """
""" Use color transforms, gradients, etc., to create a thresholded binary image. """
""" Apply a perspective transform to rectify binary image ("birds-eye view"). """
""" Detect lane pixels and fit to find the lane boundary. """
""" Determine the curvature of the lane and vehicle position with respect to center. """
""" Warp the detected lane boundaries back onto the original image. """
""" Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position. """
