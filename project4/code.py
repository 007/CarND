#!/usr/bin/env python

import cv2
import glob
import json
import numpy as np

CHESS_X = 9 # inner-horizontal corners on calibration images
CHESS_Y = 6 # inner-vertical corners on calibration images
IMG_W = 1280 # hard-code input width
IMG_H = 720 # hard-code input height

""" Wrap image reading with error handling - cv2 just returns fine if image doesn't exist """
def read_image(fname):
    image = cv2.imread(fname)
    assert image is not None, "image {} failed to load".format(fname)
    return image

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
    calibration_shape = (IMG_W,IMG_H)
    obj_points = []
    img_points = []

    objp = np.zeros((CHESS_Y*CHESS_X, 3), np.float32)
    objp [:,:2] = np.mgrid[0:CHESS_X,0:CHESS_Y].T.reshape(-1,2)
    for fname in file_array:
        print("processing {}".format(fname))
        img = read_image(fname)
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
def image_to_threshold(img, thresh_min=100,thresh_max=255):

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


def calculate_warp_params():
    border = 64
    from_shape = np.float32([ [595, 435], [690, 435], [1050, 675], [275, 675] ])
    to_shape = np.float32([ [border, border], [IMG_W-border, border], [IMG_W-border, IMG_H-border], [border, IMG_H-border] ])
    M = cv2.getPerspectiveTransform(from_shape, to_shape)
    return M

""" Apply a perspective transform to rectify binary image ("birds-eye view"). """
def perspective_warp_lane(img):
    M = calculate_warp_params()
    img_size = (IMG_W, IMG_H)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

""" invert perspective warp """
def perspective_unwarp_lane(img):
    M = calculate_warp_params()
    img_size = (IMG_W, IMG_H)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.WARP_INVERSE_MAP)
    return warped

""" Detect lane pixels and fit to find the lane boundary. """
def find_lane_lines(img):
    histogram = np.sum(img, axis=0)

    # Assuming you have created a warped binary image called "img"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[np.int(IMG_H/2):,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(IMG_W/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(IMG_H/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = IMG_H - (window+1)*window_height
        win_y_high = IMG_H - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    """ Determine the curvature of the lane and vehicle position with respect to center. """
    # calculate center
    center = (IMG_W / 2) - ((np.polyval(left_fit, IMG_H) + np.polyval(right_fit, IMG_H)) / 2)

    yspace = np.linspace(0, IMG_H - 1, num=IMG_H)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/IMG_H # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # convert center from pixels to meters
    center_m = xm_per_pix * center

    # Fit new polynomials to x,y in world space
    left_fit_meters = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_meters = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_meters[0]*IMG_H*ym_per_pix + left_fit_meters[1])**2)**1.5) / np.absolute(2*left_fit_meters[0])
    right_curverad = ((1 + (2*right_fit_meters[0]*IMG_H*ym_per_pix + right_fit_meters[1])**2)**1.5) / np.absolute(2*right_fit_meters[0])
    # Now our radius of curvature is in meters - average two sides for return
    curve = (left_curverad + right_curverad) / 2

    return left_fit, right_fit, curve, center_m


""" Warp the detected lane boundaries back onto the original image. """
def build_lane_overlay(warped, left_fit, right_fit):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()

    yspace = np.linspace(0, IMG_H - 1, num=IMG_H)
    left_points =  np.polyval(left_fit, yspace)
    right_points = np.polyval(right_fit, yspace)
    pts_left = np.array([np.transpose(np.vstack([left_points, yspace]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_points, yspace])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    # color order is RGB
    cv2.fillPoly(color_warp, np.int_([pts]), (64, 128, 255))

    # Warp the blank back to original image space
    newwarp = perspective_unwarp_lane(color_warp)
    return newwarp

""" Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position. """
def output_with_overlays(img, lane_overlay, curve, center):
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, lane_overlay, 0.3, 0)
    # TODO: add text overlay for curvature
    # TODO: add text overlay for center position
    return result

def pipeline_init():
    # skip for now
    # compute_calibration(glob.glob('./camera_cal/calibration*.jpg'))
    img = read_image('./camera_cal/calibration2.jpg')
    dst = correct_distortion(img)
#    cv2.imwrite('./output/calibration_input.jpg', img)
#    cv2.imwrite('./output/calibration_output.jpg', dst)

def pipeline(input_image):

#    cv2.imwrite('./output/traffic_input.jpg', input_image)
    corrected = correct_distortion(input_image)
#    cv2.imwrite('./output/traffic_calibrated.jpg', corrected)

    threshold = image_to_threshold(corrected)
#    cv2.imwrite('./output/traffic_thresholded.png', threshold * 255) # binary * 255 = black/white

    warped = perspective_warp_lane(threshold)
#    cv2.imwrite('./output/traffic_perspective.png', warped * 255) # binary * 255 = black/white

    left_fit, right_fit, curve, center = find_lane_lines(warped)

    overlay = build_lane_overlay(warped, left_fit, right_fit)
    final = output_with_overlays(corrected, overlay, curve, center)
    return final

def video_pipeline(input_video):
    from moviepy.editor import VideoFileClip
    in_vid = VideoFileClip(input_video)
    out_vid = in_vid.fl_image(pipeline)
    out_vid.write_videofile('./output/annoated_video.mp4', audio=False)

if __name__ == '__main__':
    pipeline_init()
#    input_image = read_image('test_images/test_fail.jpg')
#    output = pipeline(input_image)
#    cv2.imwrite('./output/final_overlay.jpg', output)

    video_pipeline('./project_video.mp4')
