#**Behavioral Cloning** 

##Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
- [ ] Use the simulator to collect data of good driving behavior
- [ ] Build, a convolution neural network in Keras that predicts steering angles from images
- [ ] Train and validate the model with a training and validation set
- [ ] Test that the model successfully drives around track one without leaving the road
- [ ] Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[recovery-animation]: ./sample/recovery-sample.gif "Recovery data animation"
[sharp-turn]: ./sample/center_2017_02_25_15_01_04_164.jpg "Normal Image"
[sharp-turn-flip]: ./sample/xenter_2017_02_25_15_01_04_164.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

### Required Files

####1. Required files for submission

My project includes the following files:
* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network
* [writeup_report.md](writeup_report.md) (this document) summarizing the results
* [video.mp4](video.mp4) displaying self-driving performance


### Quality of Code

####1. Code functionality

Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing
```bash
python drive.py model.h5
```

####2. Code usability and legibility

The `model.py` file contains the code for training and saving the convolution neural network. It uses a Python generator for training data and a `fit_generator` for the Keras input. The file contains comments to explain how the code works, and the code is factored into functions where appropriate.


###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model implements the [NVIDIA SDC architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with a few modifications.

It first employs cropping in-model, as recommended in the associated paper. I found that this does provide for a simpler workflow and faster round-trip time for experiments. I chose to use the equivalent of the [Comma.ai normalization](https://github.com/commaai/research/blob/master/train_steering_model.py#L28-L30) to a range of -1..1 instead of the standard `x / 255` range of 0..1. The model uses [3 layers](model.py#L59-L61) of 5x5 convolution, each with a stride of 2. They vary in depth, increasing from 24 to 36 to 48. Then there are [two identical layers](model.py#L63-L64), 3x3 convolution with stride 1 and depth 64.

After flattening, there are [3 fully-connected layers](model.py#L68-L70) of the form [`fc / bn / elu / drop`](model.py#L35-L38). The FC layer depths are 100, 50 and 10, respectively. [Batch normalization](https://arxiv.org/abs/1502.03167) is employed between the FC layer and the activation function to provide more consistent values. The activation function is exponential rectified linear unit (ELU) versus the usual ReLU.

Model summary below:

| Layer | Type                    | Output Shape         | Params    | Connected to              |
| ------ | ------------------------ | -------------------- | ---------- | ------------------------- |
| start | (InputLayer)              | (160, 320, 3)  | 0          |                           |
| crop | (Cropping2D)               | (70, 320, 3)   | 0          | start               |
| normalize | (Lambda)              | (70, 320, 3)   | 0          | crop                |
| dropout_0 | (Dropout)             | (70, 320, 3)   | 0          | normalize           |
| conv_5_1 | (Convolution2D)        | (35, 160, 24)  | 1824       | dropout_0           |
| conv_5_2 | (Convolution2D)        | (18, 80, 36)   | 21636      | conv_5_1            |
| conv_5_3 | (Convolution2D)        | (9, 40, 48)    | 43248      | conv_5_2            |
| conv_3_1 | (Convolution2D)        | (7, 38, 64)    | 27712      | conv_5_3            |
| conv_3_2 | (Convolution2D)        | (5, 36, 64)    | 36928      | conv_3_1            |
| flatten | (Flatten)               | (11520)        | 0          | conv_3_2            |
| dense_1 | (Dense)                 | (100)          | 1152100    | flatten             |
| batchnormalization_1 | (BatchNormalization)| (100)          | 400        | dense_1             |
| activation_1 | (Activation)       | (100)          | 0          | batchnormalization_1 |
| dropout_1 | (Dropout)             | (100)          | 0          | activation_1        |
| dense_2 | (Dense)                 | (50)           | 5050       | dropout_1           |
| batchnormalization_2 | (BatchNormalization)| (50)           | 200        | dense_2             |
| activation_2 | (Activation)       | (50)           | 0          | batchnormalization_2 |
| dropout_2 | (Dropout)             | (50)           | 0          | activation_2        |
| dense_3 | (Dense)                 | (10)           | 510        | dropout_2           |
| batchnormalization_3 | (BatchNormalization)| (10)           | 40         | dense_3             |
| activation_3 | (Activation)       | (10)           | 0          | batchnormalization_3 |
| dropout_3 | (Dropout)             | (10)           | 0          | activation_3        |
| steering_prediction | (Dense)     | (1)            | 11         | dropout_3           |

| **Trainable params** | **1,289,339** |
| :---: | :---: | :---: |
| **Non-trainable params** | **320** |
| **Total params** | **1,289,659** |

---


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. There is a dropout from the [initial input](model.py#L57), as well as [individual dropout](model.py#L38) between each FC layer.

The model was [trained](model.py#L201-L202) and [validated](model.py#L203) on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an [Adam optimizer](http://sebastianruder.com/optimizing-gradient-descent/index.html#adam). The learning rate was set much higher than the initial / default value due to use of multiple dropout layers, as recommended in the original [dropout paper](http://www.jmlr.org/papers/v15/srivastava14a.html). Since Adam is a self-tuning optimizer, this had less effect than it would have with SGD or similar.

####4. Appropriate training data

Training data was chosen with care. The orignal training data presented from the course notes had undesirable characteristics, namely it combined regular driving with intentional mistakes and correction. My training data was generated as smooth lane-centered driving, with a smaller set of augmentation data for correction _without_ the intentional mistakes.

### Architecture and Training Documentation

####1. Solution Design Documentation

####2. Model Architecture Documentation

####3. Training Dataset and Training Process Documentation

This step ended up being the **most** important part of the entire project. Regardless of my experiments in the model / architecture sections, running with "bad" data made the results much more erratic and less consistent across even small changes.

Training data was generated several ways. After the class data turned out to be less than useful, I ended up recording my own training data in four distinct batches:
 * The first used keyboard input, but ended up being too noisy with hard transitions between zero steering and over-correction.
 * I then switched to the beta simulator for the second batch, and attempted to create data with the mouse input. The data itself was smoother, but the interface was cumbersome and I ended up including a lot of mistakes due to my lack of skill with the steering controls.
 * The third batch of data was my primary training input, smooth and consistent driving with a game controller, centered in the lane for several laps in both directions around the track.
 * Batch four was my correction data, where I would drive as before, but switch recording off and back on as I made intentional mistakes and then corrected them.

With the game controller, I was able to drive efficiently and record useful data. Using a PS3 controller allowed me to get both analog input (vs digital of keyboard) and smooth and simple input (vs cumbersome mouse). The recovery dataset ended up being much more useful than the default data, since it eliminated the "make a mistake" part of the data, so the system could train only on the "correct a mistake" version.

![alt text][recovery-animation]


The recorded data was in good form, but wouldn't be sufficient for training without augmentation. The steering data would be biased to one side (whichever direction the car was going) on anything but perfect data. To eliminate this bias (and to get double the training data "for free") I added a second copy of each training sample with a horizontal flip. The steering data for the second copy was likewise negated to match. This eliminates any bias, since the average between any two samples `X` and `-X` is zero.


Sample image - steering angle `0.6961063`
![alt text][sharp-turn]

Augmented (horizontal flip) - steering angle `-0.6961063`
![alt_text][sharp-turn-flip]









###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
