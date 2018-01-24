#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./distOfclass.jpg "Visualization of classes of train dataset"
[image2]: ./grayscale.png "Grayscaling"
[image3]: ./normalization.png "Random Noise"
[image4]: ./test_signs/Stop.jpg "Traffic Sign 1"
[image5]: ./test_signs/Speed limit 60kmh.jpg "Traffic Sign 2"
[image6]: ./test_signs/Road work.jpg "Traffic Sign 3"
[image7]: ./test_signs/Turn left ahead.jpg "Traffic Sign 4"
[image8]: ./test_signs/Wild animals crossing.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data of train dataset scatters over 43 classes.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscale is one channel and easy to be converted to vectors.

Here is an example of a traffic sign image before and after gray-scaling.

![alt text][image2]

As a last step, I normalized the image data because unequal variance and non-zero dataset will improve optimization process.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following:
the mean of augmented data is 0 and variance of augmented data equal, additionally noise is added to augmented data set.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1                                       | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6                  |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs  5x5x16                  |
| Flatten               | outputs 400                                   |
| Fully Connected       | outputs 120                                   |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully Connected       | outputs 84                                    |
| RELU                  |                                               |
| Fully Connected       | outputs 43                                    |

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyper-parameters such as learning rate.

To train the model, I used an Adam-optimizer, batch size is 256, 50 epochs and learning rate 0.001, keep probability of drop-out is 0.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.96
* validation set accuracy of 0.96 
* test set accuracy of 0.937

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?I shuffled the train data set and each iteration gets a batch of samples because of random selecting.
* What were some problems with the initial architecture?Accuracy can’t go beyond 93%
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or under-fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why? Epoch is adjusted. I increased Epoch to 50. Because when it is 10, validation accuracy is only 0.89. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?A convolution layer increases the depth of features and generates the fundamental features. A dropout layer prevents from overfitting when epoch increases.Before a drop-out  
was used the validation accuracy was about 91% but after a drop out was used to connect the first fully connected layer’s activation function the accuracy increased to 96%
If a well known architecture was chosen:
* What architecture was chosen?LeNet is chosen.
* Why did you believe it would be relevant to the traffic sign application?It uses two convolutional layers to grasp the basic features of an image.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? 93.7% is the test accuracy and 4 out of 5 new images are predicted correctly.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because the sign is speed limit(60km/h) and there are lots of speed limit signs with different digits. Some of digits look like similar, for example 60 with 80, or 60 with 90.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign             | Stop Sign                          		| 
| Speed limit (60km/h)  | Road work                          		|
| Road work             | Road work                                     |
| Turn left ahead       | Turn left ahead                               |
| Wild animals crossing	| Wild animals crossing                         |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.7%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the soft-max probabilities for each prediction. Provide the top 5 soft-max probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the I-python notebook.

For the first image, the model is relatively sure that this is a Stop-sign (probability of 0.91), and the image does contain a stop sign. This Stop sign is pictured not from an angle and not jitter, nor is there objects in background. The only difficulty is alike some speed limit signs.

| Probability         	|     Prediction	        		 
|:---------------------:|:---------------------------------------------:| 
| .91                   | Stop                          		| 
| .08                   | Speed limit (30km/h)                          |
| .01                   | Speed limit (80km/h)                          |
| .00                   | Yield                                    	|
| .00                   | Speed limit (30km/h)                          |


For the second image, the model recognizes it as Road work wrongly.This image is difficult to tell apart the digits of other speed limit. Additionally, it has white background and black digits which were very similar as Road work’s logos. Black and white are not affected when from color image to grayscale.
| Probability         	|     Prediction	        		 
|:---------------------:|:---------------------------------------------:| 
| .99                   | Road work                          		| 
| .00                   | Speed limit (50km/h)                          |
| .00                   | Speed limit (30km/h)                          |
| .00                   | Speed limit (60km/h)                          |
| .00                   | Keep right                         		|

For the third image, the model is sure it is Road work.This image has jitter and might be difficult to classify.   
| Probability         	|     Prediction	        		 
|:---------------------:|:---------------------------------------------:| 
| 1.00                  | Road work                                     | 
|  .00                  | General caution				|
|  .00                  | Dangerous curve to the right                  |
|  .00                  | Right-of-way at the next intersection         |
|  .00                  | Go straight or right				|

For the fourth image, the model is quite sure it is turn left ahead.The quality of this image is good.Except that the background is a circle which exists in many signs. 
| Probability         	|     Prediction	        		 
|:---------------------:|:---------------------------------------------:| 
| 1.00                  | Turn left ahead                               | 
|  .00                  | Speed limit (30km/h)                          |
|  .00                  | Roundabout mandatory                          |
|  .00                  | Ahead only                         		|
|  .00                  | Keep right					|

For the fifth image, the model is quite sure it is wild animals crossing. Wild animals crossing image has jitter and might be difficult to classify 
| Probability         	|     Prediction	        		 
|:---------------------:|:---------------------------------------------:| 
| 1.00                  | Wild animals crossing                         | 
|  .00                  | Road work                                  	|
|  .00                  | Slippery road					|
|  .00                  | Speed limit (50km/h)         			|
|  .00                  | Double curve                                 	|

### (Optional) Visualizing the Neural Network (See Step 4 of the I-python notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

