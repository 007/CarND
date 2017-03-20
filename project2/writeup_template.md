#**Traffic Sign Recognition**

## Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](carnd_rmoore_project_02.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.

I use generic `python` functions and data manipulation for the summary statistics:
* The size of training set is 39,209 examples
* The size of test set is 126,630 examples
* The shape of a traffic sign image is 32x32 with a depth of 3 channels
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in cells 3 through 7 of the IPython notebook.

Cell 4 shows a sample of random images from the training set. It provides a sample of the original image (left) along with a contrast-maximized version (right) courtesy of the `max_contrast` function. I added this visualization after seeing samples such as the first, where the sign was barely visible for a human.

Cell 5 shows the frequency of each of the 43 labels, with the highest frequency of 2250 and the lowest of 210. It is clear from the chart in cell 6 that the dataset is not well-balanced, as the number of examples for 10 labels are above 1500 each, and the number for another 10 or more have less than 500 each.

Cell 7 is a direct textual listing of these facts, and shows min/max frequency and the ratio between the two - the highest frequency sign occurs 10.7x as often as the lowest-frequency sign! It also shows the individual frequencies, with sign type 0, 19, 37 having 210 examples and sign type 1 and 2 having 2220 and 2250 respectively.


### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in cell 4, as described above. The only processing on image data before training is histogram equalization on the `L` channel of a `LAB` image after converting from `RGB`. For both training and testing, data is processed to maximize histogram separation. [Cireșan, et al](http://people.idsia.ch/~juergen/nn2012traffic.pdf) suggests a similar approach, but with additional local histogram equalization and contrast enhancement.




#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)


Training and testing data is defined in cell 1, and validation data is defined in cell 12.

For each epoch, the training data was split into a validation and training set. `VALIDATION_SET_PERCENT` is used to determine the size (10%), and `sklearn.shuffle` was performed on the training data set before splitting into train and validate.

Since the training data was 39,209 examples, 10% is 3,920.

To augment the data, cell 9 implements 8 rotations (+/- 10 degrees in 2 degree increments) and 16 translations (+/- 4 pixels in each direction). Each item is multiplicative - rotations are performed to augment the data by 8x, and then all 8 are subject to 16 translations each. This is substantially similar to [Howard](https://ai2-s2-pdfs.s3.amazonaws.com/d671/75d17c450ab0ac9c256103828f9e9a0acb85.pdf) for the initial image processing step before subsequent scaling and cropping. There is additional code to add random noise, but for some reason it ended up causing some kind of feedback where it would add noise to noise to noise until the images were useless.

With just 8 rotations and 16 (multiplicative) translations the data is augmented to 128 output samples for each input sample. Augmented images can be seen below cell 10. The second image is the easiest to tell for rotation, where the bottom of the triangle is tilted to the left. The third image is easiest to see the translation, where the progression betwen the 2nd and 3rd images in the row shows a smaller black band on the right side.

My final training set had 90% of the training data (35,289) augmented by 128x for a total of 4,516,992 training images. My validation set, as before, contains 10% of the initial training data, 3,920 images. The test set was fixed at 12,630 images from the `test.p` source.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture is defined in the `LeNet` function in cell 8. Helper functions are included to handle layer sizing for inputs and outputs - it would have been much nicer to learn and use Keras in place of these.

The architecture is essentially the LeNet architecture from the previous project.
* First layer is a 5x5 convolutional layer with 6 outputs and `relu` activation
* Second layer is `maxpool` of 2 with stride 1
* Third layer is another 5x5 convolutional layer with 16 outputs and `relu` activation
* Fourth layer is another `maxpool` of 2 with stride 1
* Fifth layer is fully connected with 120 outputs and `relu` activation
* Sixth layer is fully connected with 84 outputs and `relu` activation
* Seventh layer is the classification layer, fully connected with `n_classes=43` outputs

Dropout is added after flattening (layer 4) and between FC layers 5, 6 and 7 for training. There is no activation function for the classification layer.


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in cell 12. There is some setup for some evaluation functions, learning rate params, optimizer and all of the necessary placeholders in cell 11.

I selected the same optimizer that was used in the previous project. I confirmed that it was the best for this particular use-case by referencing [Sebastian Ruder's optimization page](http://sebastianruder.com/optimizing-gradient-descent/index.html#visualizationofalgorithms) which recommends `Adam` as one of the best options, and particularly for our use case it should work well.

Batch size is set at 128, but that means a slice of 128 of the original training images. Those are expanded by 128x to result in actual batch sizes of `128 x 128 = 16384`. 100 epochs were experimented with, and the best result was saved at each epoch as well as the final model. Early cutoff happens when the validation accuracy goes above 99%, usually after 30-40 epochs. Each epoch took about 2 minutes on a 1070 GPU with `tensorflow-gpu` and `cuda` / `cudnn` libraries installed. The hyperparameters are partly defined in the architecture above, where FC layers of 120 and 84 were arbitrary. The learning rate was left at 0.001 since it seemed sufficient to make progress without bouncing across the gradient minima. The only other parameters are for dropout (20%) and validation set size (10%).


#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located with the training code in cell 12. It runs both validation and test after each epoch, and saves and restores the model state around both to ensure no "cheating" where the model sees / updates itself for the validation or testing data.

My final model results were:
* validation set accuracy of 99.2
* test set accuracy of 95.3

The majority of the solution was lifted from project 1, since the LeNet architecture is very well-suited to classifying a set of images into categories. I researched some of the individual steps as referenced above, looking for ways to normalize and expand the testing and training data. The biggest trial-and-error parts of this project were working with Python and OpenCV to get data into the right formats in the right places. After the basic implementation was already complete I discovered the [CIFAR dataset and reference model](https://www.tensorflow.org/tutorials/deep_cnn/), which would have been faster for multi-GPU training, and may have resulted in more accurate classification.

Learning rate and batch size were the parameters that were tuned. Dropout was added after initial review, and improved the test performance significantly. As above, the hyperparameters were essentially defined by the choice of LeNet, and good performance came without any changes to the number or size of layers.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.


Seven different signs were found on the web, you can see the data in cell 13. Each file is named with its numeric and text category, i.e. `sample-1-speedlimit30.png` is a category 1 sign, which corresponds to `speed limit 30`.

The fact that most of the images are idealized could go either way. They might be too normal for the classifier which expects some noise and hasn't seen these as inputs, but I would expect that the deep and convolutional nature of the classifier would see the image features vs expecting detailed data. In all cases having a static, high-resolution image that's being downsampled to 32x32 is very different from the training data. Even the camera of the original might be a contributing factor, in that the model is trained on whatever imperfections or patterns may have existed there.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image           |  Prediction      |
|:--------------:|:------------------:|
| Speed limit 30 | Speed Limit 60     |
| Stop           | No Entry           |
| Speed Limit 50 | Speed Limit 60     |
| Pedestrians    | Right-of-Way       |
| Bicycles       | Right-of-Way       |
| Wild Animal    | Wild Animal        |
| Keep Right     | Beware of Ice/Snow |


The model was able to correctly guess 1 of the 7 traffic signs, which gives an accuracy of 14%. This is awful compared to the validation and test set, indicating that either the model was overfit or that the images are significantly different enough that the classifier can't recognize them accurately.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in cell 14, and the `top_k` visualization is show in cell 15.

For the one image that does match (Wild Animal), certainty is 1.86.

For the rest of the images, one out of the remaining 6 shows a `top_k = 5` match for the correct category, at a certainty of -1.36 and behind a certainty of 3.00. This shows that the model is all over the place - it's very certain in some cases when it's wrong, and very uncertain in other cases when it's correct. Even when running `top_k = 43` the results are nearly useless - the maximum confidence is something around 3, and most of the time the top 5 are either negative or &lt; 1.0.

It gets some images in the right family (`Speed Limit 30` and `Speed Limit 50` are both classified as `Speed Limit 60`), but in the wrong category. Balancing the training data may help with that, since there were several hundred more of some categories even within the same families.

The best way to improve this would be to augment the training set with additional non-normalized static images, providing more real-life snapshots versus video frames. It would also be interesting to try various resizing algorithms to see if any of them make a difference - I would imagine that smoothing the reductions would help, versus the existing images which are mostly nearest-neighbor or linear sampling. It would also help to extract new testing data from similar corpora to the original, something like extracting off-angle real-world signs from Google Maps instead of idealized stock photography samples.
