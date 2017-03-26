## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[bbox-sample]: ./output/small-boxes.png
[bbox-saturated]: ./output/saturated.png
[heatmap]: ./output/heatmap.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG features are extracted in [train.py](train.py#L10) using the [`process_image_hog`](hog.py#L25-L32) function for image normalization and color correction, which calls [`get_hog_features`](hog.py#L14-L23) to do the actual HOG extraction. It gets back a feature vector (`feature_vector=True`) versus a raw gradient map since training content is windowed to 64x64 for all images. The `process_image_hog` function currently has scaling and normalization built in, but was [originally returning image features directly](https://github.com/007/CarND/blob/1b2a7926f7c179c3482ad8a6bde728d2216782b7/project5/hog.py) during the training runs - it has subsequently been refactored.

I started by reading in all the `vehicle` and `non-vehicle` images and [extracting HOG features from each](train.py#L21-L25). The full set of features from both classes is concatenated to create a single feature vector, and a set of `1`s and `0`s of equal length to the `vehicle` and `non-vehicle` datasets (respectively) is concatenated to create a single label vector.

#### 2. Explain how you settled on your final choice of HOG parameters.

I started with the parameters proscribed in the class docs, and experimented from there to find better options. I read a [presentation](http://slideplayer.com/slide/5111258/) suggesting that a different set of parameters were useful for the same general problem solved in this project, so I implemented their choice of parameters. After that, I ran [a set of experiments](experiment.py) to see which other parameters I should change, specifically with regard to color space and the `C` parameter for SVM smoothness bounds.

```
HLS C = 1.0 accuracy is 98.592 , false positives 29.0 , false negatives 21.0 , matches 3502.0 out of 3552
HLS C = 0.8 accuracy is 98.592 , false positives 29.0 , false negatives 21.0 , matches 3502.0 out of 3552
HLS C = 0.5 accuracy is 98.592 , false positives 29.0 , false negatives 21.0 , matches 3502.0 out of 3552
HLS C = 0.2 accuracy is 98.592 , false positives 29.0 , false negatives 21.0 , matches 3502.0 out of 3552
HLS C = 0.1 accuracy is 98.592 , false positives 29.0 , false negatives 21.0 , matches 3502.0 out of 3552
HSV C = 1.0 accuracy is 98.452 , false positives 28.0 , false negatives 27.0 , matches 3497.0 out of 3552
HSV C = 0.8 accuracy is 98.452 , false positives 28.0 , false negatives 27.0 , matches 3497.0 out of 3552
HSV C = 0.5 accuracy is 98.452 , false positives 28.0 , false negatives 27.0 , matches 3497.0 out of 3552
HSV C = 0.2 accuracy is 98.452 , false positives 28.0 , false negatives 27.0 , matches 3497.0 out of 3552
HSV C = 0.1 accuracy is 98.452 , false positives 28.0 , false negatives 27.0 , matches 3497.0 out of 3552
LAB C = 1.0 accuracy is 98.874 , false positives 22.0 , false negatives 18.0 , matches 3512.0 out of 3552
LAB C = 0.8 accuracy is 98.874 , false positives 22.0 , false negatives 18.0 , matches 3512.0 out of 3552
LAB C = 0.5 accuracy is 98.874 , false positives 22.0 , false negatives 18.0 , matches 3512.0 out of 3552
LAB C = 0.2 accuracy is 98.874 , false positives 22.0 , false negatives 18.0 , matches 3512.0 out of 3552
LAB C = 0.1 accuracy is 98.874 , false positives 22.0 , false negatives 18.0 , matches 3512.0 out of 3552
LUV C = 1.0 accuracy is 98.789 , false positives 23.0 , false negatives 20.0 , matches 3509.0 out of 3552
LUV C = 0.8 accuracy is 98.789 , false positives 23.0 , false negatives 20.0 , matches 3509.0 out of 3552
LUV C = 0.5 accuracy is 98.789 , false positives 23.0 , false negatives 20.0 , matches 3509.0 out of 3552
LUV C = 0.2 accuracy is 98.789 , false positives 23.0 , false negatives 20.0 , matches 3509.0 out of 3552
LUV C = 0.1 accuracy is 98.789 , false positives 23.0 , false negatives 20.0 , matches 3509.0 out of 3552
RGB C = 1.0 accuracy is 96.256 , false positives 61.0 , false negatives 72.0 , matches 3419.0 out of 3552
RGB C = 0.8 accuracy is 96.256 , false positives 61.0 , false negatives 72.0 , matches 3419.0 out of 3552
RGB C = 0.5 accuracy is 96.256 , false positives 61.0 , false negatives 72.0 , matches 3419.0 out of 3552
RGB C = 0.2 accuracy is 96.256 , false positives 61.0 , false negatives 72.0 , matches 3419.0 out of 3552
RGB C = 0.1 accuracy is 96.256 , false positives 61.0 , false negatives 72.0 , matches 3419.0 out of 3552
XYZ C = 1.0 accuracy is 95.946 , false positives 63.0 , false negatives 81.0 , matches 3408.0 out of 3552
XYZ C = 0.8 accuracy is 95.946 , false positives 63.0 , false negatives 81.0 , matches 3408.0 out of 3552
XYZ C = 0.5 accuracy is 95.946 , false positives 63.0 , false negatives 81.0 , matches 3408.0 out of 3552
XYZ C = 0.2 accuracy is 95.946 , false positives 63.0 , false negatives 81.0 , matches 3408.0 out of 3552
XYZ C = 0.1 accuracy is 95.946 , false positives 63.0 , false negatives 81.0 , matches 3408.0 out of 3552
YCrCb C = 1.0 accuracy is 98.677 , false positives 24.0 , false negatives 23.0 , matches 3505.0 out of 3552
YCrCb C = 0.8 accuracy is 98.677 , false positives 24.0 , false negatives 23.0 , matches 3505.0 out of 3552
YCrCb C = 0.5 accuracy is 98.677 , false positives 24.0 , false negatives 23.0 , matches 3505.0 out of 3552
YCrCb C = 0.2 accuracy is 98.677 , false positives 24.0 , false negatives 23.0 , matches 3505.0 out of 3552
YCrCb C = 0.1 accuracy is 98.677 , false positives 24.0 , false negatives 23.0 , matches 3505.0 out of 3552
YUV C = 1.0 accuracy is 99.099 , false positives 16.0 , false negatives 16.0 , matches 3520.0 out of 3552
YUV C = 0.8 accuracy is 99.099 , false positives 16.0 , false negatives 16.0 , matches 3520.0 out of 3552
YUV C = 0.5 accuracy is 99.099 , false positives 16.0 , false negatives 16.0 , matches 3520.0 out of 3552
YUV C = 0.2 accuracy is 99.099 , false positives 16.0 , false negatives 16.0 , matches 3520.0 out of 3552
YUV C = 0.1 accuracy is 99.099 , false positives 16.0 , false negatives 16.0 , matches 3520.0 out of 3552
```
I noted not just accuracy, but false positives (classifier says YES when answer is NO) and false negatives (classifier says NO when answer is YES). Focusing my methodology on reducing false positives by as much as possible helped with future steps by reducing the total amount of filtering and subsequent data processing to successfully discard such incorrect data.

The only surprising result was that there was **zero** difference for my choices of `C` parameters across all colorspaces. I intend to redo the analysis after completing this project to see whether there was a problem with my approach (most likely), or whether this dataset was regular enough that the SVM boundary conditions were smooth to begin with.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The [SVM training code](train.py#L38-L54) is part of [`train.py`](train.py). It's really just the two lines 39 and 40 to implement and train, but the other supporting code is used to calculate false positive/negative counts based on the difference between the `predict` output from the SVM and the ground-truth labels from `y_test`. Each feature vector was approximately 30k, but the `scikit` `LinearSVC` code handled it exceptionally well.

In the [final image detection code](vid.py), features are [extracted](vid.py#L83-L84) and [normalized](vid.py#L85) to match the requirement of the [saved SVM](vid.py#L19-L25).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The search code is implemented in [`vid.py`](vid.py#L61-L93). After cropping and converting the input image to the appropriate colorspace, the entire image is passed through the `process_image_hog` function to extract features, this time with `feature_vector=False`. An exhaustive search is performed over the image area for each frame. Experimentally, I decided that no scaling was required, since I got acceptable feature extraction with a single scale and lots of overlap.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The first image shows some individual detected bounding boxes near the center of the image, as well as some massively-overlapping boxes on the car to the right.

![alt text][bbox-sample]

This is a second image where both vehicles are completely saturated with detection triggers.

![alt text][bbox-saturated]

This is a sample of the [heatmap](heatmap.py) that I generated from the overlap of bounding boxes as above.

![alt text][heatmap]

I also have a heatmap of [the full video](mini_heatmap.mp4) available.

Starting from the feature selection, I optimized my selection criteria for fewer false positives, at the expense (possibly) of false negatives. The choice of 3-channel YUV HOG features gave the best SVM performance on the input training/test data, and I experimented to confirm that that success would carry over to the project output. Using a [high threshold](bounding.py#L20) of 10 overlapping detection windows for the heat map input and [exponential falloff](bounding.py#L34) for frame history gives a more stable feature set. Using a ["double buffer"](bounding.py#L31-L39) helped as well, since only features that crossed the original threshold and a second (cumulative) history-based threshold were used for final bounding box detection.


---

###  Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./annotated_video.mp4). The vehicles are indicated clearly and accurately most of the time, and there are a few brief false positives around road signs.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

*This paragraph was filler text from the writeup template, but it was exactly my approach, so I added links to the relevant code*

I [recorded the positions of positive detections](vid.py#L29) in each frame of the video.  From the positive detections I created a [heatmap](bounding.py#L12-L17) and then [thresholded that map](bounding.py#L18-L21) to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to [identify individual blobs](bounding.py#L43) in the heatmap.  I then assumed each blob corresponded to a vehicle.  I [constructed bounding boxes](bounding.py#L49-L58) to cover the area of each [blob detected](bounding.py#L47-L48).

False positives were filtered with [thresholding and history](bounding.py#L31-L39), making sure that several feature hits were required successively over a period of time before a feature / set of features would be considered a vehicle.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The 2 biggest problems I faced were calculating the HOG feature sizes to be able to extract them from a single-pass version of the image, and suppressing false positives. It took a lot of debugging lines and re-reading the documentation for HOG to understand [what shape the feature blocks would end up](vid.py#L13). Getting the number of false positives down from crazy-alot to just a few was also quite difficult, and involved a lot of experimentation across the entire video to account for frame history and decay parameters.

The pipeline, as implemented, is too slow for real-world use. The first thing to optimize would be the feature extraction code, primarily to see if fewer features would still give acceptable performance, but also to speed it up and make it run closer to real-time. Because of this limitation in my original implementation, I ended up breaking out several steps of the pipeline to their own files and duplicating some code. It helped with experimentation to be able to `pickle` and store one step and pick up from there with an entirely new file, but may make it harder to review - sorry about that!

Adding more (or better) history tracking would probably help to reduce false positives, and averaging or other weighting might help to reduce false negatives. The bounding box code only takes into account the thresholded / time-tracked features, so it doesn't track and box vehicles directly, only their "shadows" over time. I would like to add image segmentation around the vehicle detection to make a better show of the detection - something like converting the image to grayscale but allowing the detected vehicles to show up in color.

I haven't experimented with any other video sources, but I would imagine that this system would perform poorly in urban conditions. Being able to detect a hard edge of a car versus a rolling hill is much easier than differentiating the windows on a building from the windows on a truck or bus. It would also probably fail at night, since the illumination conditions are so much different than the existing video and training data.
