##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[calin]: ./output/calibration_input.jpg "Raw input"
[calout]: ./output/calibration_output.jpg "After calibration"
[traffic-in]: ./output/traffic_input.jpg "Sample lane image"
[traffic-cal]: ./output/traffic_calibrated.jpg "Calibrated lane image"
[traffic-roi]: ./output/traffic_roi.jpg "Region of interest"
[traffic-thresh]: ./output/traffic_thresholded.png "Lines found"
[traffic-warp]: ./output/traffic_perspective.png "Warped projection"
[traffic-final]: ./output/final_overlay.jpg "Final output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Distortion calibration and correction code can be found on [line 33 to 73 of code.py](code.py#L33-L73).

Given a set of calibration images, "Chessboard corners" are used to calculate perspective and distortion coefficients for different sections of the frame. Each image is converted to grayscale, then `cv2.findChessboardCorners` is performed. The set of corners found is added to a list, and associated with a uniform grid - the found corners will be set to match the uniform grid after correction.

After all calibration images have been searched, the full set of found corners and matching uniform corners are fed into `cv2.calibrateCamera` to calculate the calibration matrix and distortion parameters. Those parameters are saved as a JSON file.

Sample input image:
![alt text][calin]

Calibrated output image:
![alt text][calout]

### Pipeline (single images)

The pipeline is defined simply in [`code.py` lines 253 to 269](code.py#L253-L269), with all of the hard work in the functions above.

#### 1. Provide an example of a distortion-corrected image.
The images above are an example of distortion-corrected images, but I assume that this is meant to show a working example.

The pipeline images will be based on this input image:
![alt text][traffic-in]

The first step is to correct for distortion using the calibration data calculated as above. This happens on [line 256 via the `correct_distortion` function](code.py#L256).
Distortion corrected:
![alt text][traffic-cal]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
Thresholding is performed in the [`image_to_threshold` function](code.py#L75-L94).

The image is converted to `HLS` colorspace, and the `S` channel is used exclusively. A Sobel operator is run over the image in `x` to find mostly-vertical lines. The gradient is normalized via `abs`, then scaled from a possible range of `0 - 1` to a range of `0 - 255` based on the actual maximum value in the image. The result is filtered for a minimum value of `12`. It is also filtered for a maximum value of `255`, but that corresponds to the maximum possible value.

![alt text][traffic-thresh]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform is calculated and performed in [`perspective_warp_lane` and `calculate_warp_params`](code.py#L97-L109).

The region of interest for the perspective transform is defined in [`ROI_SHAPE`](code.py#L20-L25) as symmetric around the horizontal midpoint of the image, with a top and bottom horizontal line. Parameters are hard-coded, and were determined experimentally, with manual sanity checking for approximately correct perspective.

The [destination mapping](code.py#L100) for the perspective transform is also hard-coded as an absolute offset (`border`) from the side of the image.

Perspective area:
![alt text][traffic-roi]

Warped (thresholded) image:
![alt text][traffic-warp]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After thresholding and perspective transform, lane pixels are identified and turned into left and right lines in [`find_lane_lines`](code.py#L118-L185). The code is nearly a direct copy of the sample code from the course notes, only a parameter name or two have been changed. After finding histogram peaks for left and right groupings, a  windowing function which is used to find potential lane marker pixel clusters. It is iterated starting at the `current` location over a `margin` sized box, and points which are nonzero are counted for each window. Those points are appended to the lane indices, and if there are more than `minpix` within a window, their mean is calculated as the new `current` location for the next window.

The left and right lane lines are calculated via 2nd-order polynomial on [lines 184 and 185](code.py#L184-L185).

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

[Radius of curvature](code.py#L188-L207) is calculated via the formulae provided in the course notes - as with the windowing code, it is used almost verbatim. [Lane position](code.py#L190-L198) uses the same parameters for converting pixel space to meter space. The lane center (average of left and right fit lines at the bottom of the image) is subtracted from the image center to calculate a pixel offset, which is multiplied by the conversion constant in the `x` dimension.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Plotting the lane image back on as an overlay is performed in [`build_lane_overlay`](code.py#L212-L233). It takes the left and right poly lines and fills the space between them with a color block. Since this is performed on a perspective-warped image, it also calls `perspective_unwarp_lane` to invert the perspective transform before returning the lane overlay.

The final step in the pipeline is to composite the original image, the lane overlay, and annotations for curvature and lane position. All of these happen in [`output_with_overlays`](code.py#L235-L243), where the lane overlay is combined with a `0.3` weight on top of the original distortion-corrected image. Text is added via `cv2.putText` in the top-left corner, and the image is returned for the pipeline to continue.
![alt text][traffic-final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./annotated_video.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part of the implementation was getting a thresholding scheme that would work consistently across different images. The sample data was very helpful, in that it provided most of the combinations that would cause trouble for different steps in the thresholding function. Having straight lines, curved lines, shady sections and other cars on the road all tripped up my implementation at one point or another, and having sample data to work with as a single image made experimenting much faster.

Color space still seems to be a problem. Even working strictly with `cv2` and `imread` / `imwrite` doesn't seem to solve the colorspace mixups, since `moviepy` and `matplot` both seem to work in `RGB` and `cv2` / `numpy` / `matplot` images don't carry a notion of field ordering or identity. Most of the time this manifested as weird sky colors, but in at least one instance I ended up with the overlay data being strange colors while the background image was fine.

This pipeline will probably fail for any sharper curves than I accounted for. The ROI for the perspective transform is as small as it could be while completing the sample video, so it would likely lose track of some curves outside of its mask area. For the same reason, any samples where the vehicle is significantly off-center would probably fail.

I stopped experimenting with thresholding algorithms when I got something that was good enough, that would probably be an excellent place to look for further improvements. I would also profile the entire pipeline, it was only getting 10-20FPS even on a multi-core multi-gigahertz system. For simple experiments that's not a problem, but for real-time driving control that could potentially be disasterous. From a quick analysis it seems like the camera distortion correction is the biggest bottleneck, so downsampling or other data reduction techniques may be the only way to improve that performance. Since such an action could drastically affect the quality of lane detection, it makes sense to profile all of the functions in the pipeline more thoroughly before changing something so integral to the process.

The only thing that surprised me in completing this project was that I didn't need to implement the memory functions suggested in the course notes. I was able to get stable lane lines with no temporal averaging, and the lane detection windowing function was fast enough that I didn't have to optimize re-using the previous detection. 
