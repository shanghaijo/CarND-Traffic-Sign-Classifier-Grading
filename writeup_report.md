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
r* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./60.png "Traffic Sign 1"
[image2]: ./noway.png "Traffic Sign 2"
[image3]: ./nouturn.png "Traffic Sign 3"
[image4]: ./stop.png "Traffic Sign 4"
[image5]: ./danger_left.png "Traffic Sign 5"
[image10]: ./b4norm.png "After rotation and offset"
[image11]: ./afternorm.png "After rotation and offset"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

t####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shanghaijo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set was 34799 at the beginning
* The size of test set is 12630 images
* The shape of a traffic sign image is 32 per 32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

I didn't do any visualization of the dataset as it is quite straight forward

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the second code cell of the IPython notebook.

As a first step, I changed the images in quite a few steps:
I first rotated all the images by a random angle of +/-2° and I added these results to the training set.
I then modified all the images of the training set to offset them randomly by +/-2 pixels and added that to the training set again. I thus ended up with 139,196 images for the training set.
I planned on changing them to grayscale however, I got an error with the shuffle function and couldn't immediately figure it out.

Here is an example of a traffic sign image after offesting and rotation


![alt text][image10]
As a last step, I normalized the image data because it helps for the training of the network. Before I normalized, I tried and couldn't get over the 93% point


![alt text][image11]

My final training set had 139,196 number of images. 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
 
The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
e| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, Valid padding, outputs 10x10x16					|
| RELU     |            |
| Max Pooling         | 2x2 strides  | 5x5x16 |
|Flatten              | output 400
| Fully connected		| output 200        									|
| RELU			|        									|
|	Fully connected					|	 output 84											|
|	RELU					|												|
| Fully connected | output 43 |
h 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the ninth cell of the ipython notebook. 

To train the model, I used an Adam optimizer and an L2 regularization. I used the classic hyperparameters I found in literature like the training rate of 0.001 and beta of 0.001. I tried different values for those but couldn't really see an improvement and I thus kept those.
I trained the model through 200 epochs, it is quite a lot especially since I have 136 thousands images however, that seemed to be the best score I could get. I actually had dropouts in the network at the beginning but I finally abandonned this way since I had no overfitting to my training data.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eleventh cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.955 
* test set accuracy of 0.941



If a well known architecture was chosen:
* I chose a LeNet, I read a few articles on the internet and it seemed like a good idea to start with this one and try to finetune it. At the beginning, I got roughly 0.91 accurcy on the validation set. I ran the training and the validations and tests several times and always got results over .94 after I added the modified images to the training set.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

I was afraid the images might be dificult to be recognized because they are actually very clear and very big taking the whole frame. I was afraid that would influence quite badly

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the twelveth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| no way					| no way											|
| 60 km/h	      		| 60km/h					 				|
| Dangerous left turn			| Dangerous left turn      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.946.
At the beginning, I had an accuracy of only 4 over 5 and the speed limit was always the one not working. I actually found out that my normalization was wrong and was creating dissues.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
