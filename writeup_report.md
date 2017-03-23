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
[image6]: ./real_stop1.png "Real Stop Sign 1"
[image7]: ./real_stop2.png "Real Stop Sign 2"
[image8]: ./real_stop4.png "Real Stop Sign 3"
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
igns data set:
* The size of training set was 34799 at the beginning
* The size of test set is 12630 images
* The shape of a traffic sign image is 32 per 32
* The number of unique classes/labels in the data set is 43


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the second code cell of the IPython notebook.

As a first step, I changed the images in quite a few steps:
I first rotated all the images by a random angle of +/-2° and I added these results to the training set.
I then modified all the images of the training set to offset them randomly by +/-2 pixels and added that to the training set again. I thus ended up with 139,196 images for the training set.
I planned on changing them to grayscale however, I got an error with the shuffle function and couldn't timely figure it out.

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
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, Valid padding, outputs 10x10x16					|
| RELU     |            |
| Max Pooling         | 2x2 strides  | 5x5x16 |
|Flatten              | output 400
| Fully connected		| output 200        									|
| RELU			|        									|
|	Fully connected					|	 output 84											|
|	RELU					|												|
| Fully connected | output 43 |
 


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

I firstly used images from road rules manuals as can be seen on the images over there and I got an accuracy of 100%. I did it this way to check if, as trained on a "real life" pictures, it would be able to recognize something "cleaner".
I also tried with more real images and I got a pass each time:
![alt text][image6] ![alt text][image7]

I tried with something more challenging:
![alt text][image8]
and there I got a wrong prediction

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| no way					| no way											|
| 60 km/h	      		| 60km/h					 				|
| Dangerous left turn			| Dangerous left turn      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.946.
At the beginning, I had an accuracy of only 4 over 5 and the speed limit was always the one not working. I actually found out that my normalization was wrong and was creating issues.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last cell of the Ipython notebook.

I actually have very high probabilities, for the images I used, I am between 0.97 and 1.0! I have thus a very opinionated network. I think that the fact that I am going through 200 epochs makes it extremely sure. However, this is also true when the prediction is wrong. With the image of the stop being only partial on the image, i got a 1. probability on the wrong prediction!! 

Next steps and conclusion:
I unfortunately spent a significant amount of time figuring out the images pre-processing since this is something I never came across before I actually understood that tf had some libraries for that. I actually started out with the hyperparameters and got completely stuck at 91% accuracy, I went at that stage to gray scale but didn't get any improvements so I dropped this idea.
I experienced also with dropouts since I wanted to have a more robust network however I ended up with lower accuracy on the validation set. So it does what it is supposed to do, prevent overfitting but when there's is none, it lowers accuracy, at least in this case.
The major breakout in the accuracy level was by adding modified images to the original, add L2 regularization and go over 150 epochs.
I know understand that I went through a lot of dead ends during the course of this project and I am not through with it but I will need more time to tweak it. That would be:
- Have a better pre-processing of the images, this one is too basic, add grayscale, zoom and bigger offsets
- Limit the numbers of epochs (50 max) and still get a good accuracy by working on the hyper parameters. I think that would help prevent a very opinionated network where predictions are very binary
- The structure of the network looks ok, it is a Lenet after all, I would at most change some layers sizes but not much
- Choose better images from the web to find out how my network reacts to real life challenges.

In conclusion, this network is far from showing a great result, however, it provided me with a lot of lessons learned that I will be able to apply for the future and I will keep upgrade it during the rest of the course to see where it leads me.

