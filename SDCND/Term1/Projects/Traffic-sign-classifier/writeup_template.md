#**Traffic Sign Recognition** 

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jaychokshi/Udacity/tree/master/SDCND/Term1/Projects/Traffic-sign-classifier/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 6th and 9th code cells of the IPython notebook.  


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques?

The code for this step is contained in the 11th and 12th code cells of the IPython notebook.

I performed below three steps to preprocess the data,
1. Since subset of the data samples were blurry and on low side of the contrast, I thought to use the CLAHe method to bring about the edges better.
2. Applied Grayscaling on the image to reduce the amount of data model has to optimize for. This seems to be a popular method as others[1] also cited to be using it for similar model definition
3. As a last step, I normalized the image data between .1 and .9. It was mentioned in the lecture notes as well. This allows to remove any feature specific bias [2] from the equation.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)


I did not have to split the training data as pickle already provide validation set separately along with test set.
Though, I observed that a subset of the class image samples were considerably low compared to other classes. This may potentially create learning bias toward "striving" classes. Thus, I generated augmented images. The process of augmentation [3] was radomized where copies are made with changes include rotation, translation, and shear in the subject image.

My final training set had 51690 number of images (delta to orignal training set: 16891). My validation set and test set had 4410 and 12630 number of images.

The seventh code cell of the IPython notebook contains the code for augmenting the data set. 

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 13th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, output 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, output 1x1x400     |
| RELU					|												|
| Flatten & concatenate | 1x1x400 to 400, 5x5x16 -> 400, stitch to 800  |
| Dropout				| keep_prob: 0.5								|
| FC softmax			| 800 to 43										|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the Adam optimizer (the one in LeNet lab) with below settings, 
batch size: 100
epochs: 60
learning rate: 0.0009
mu: 0
sigma: 0.1
dropout keep probability: 0.5

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 18th and 19th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 96.1%
* test set accuracy of 95.6%

The model is adapted from LeCun classifier [1]. The initial base mode that was covered in the LeNet lab provided accuracy around 85%, so that was kind of underfitting. On the current architecture, with low batch size, learning rate, and dropout, I observed that the validation accuracy was almost 99%. While that was satisfying, I dropped that settings to reduce any overfitting issues.
With above paramters, I see that both validation and test accuracy are fairly close to each other, that proves that model is not biased towards any feature sets.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Six German traffic signs that I found on the web are displayed in cell 23rd in notebook.

The right and left turn images has totally contrasting colors, so I thought they may pose interesting challange for model.
Also, I included two different images for the same class, to test if how model performs.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Left turn      		| Ahead only   									| 
| Right of way			| Right of way									|
| Yeild 	      		| Yeild											|
| keep right			| Keep right      								|
| Road work 			| Road work	      							|
| Road work 			| Speed limit 70      							|

The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.6%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Top five soft max probabilities:
1. Left turn image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Ahead only   									| 
| .0     				| Speed limit (20km/h) 							|
| .0					| Roundabout mandatory							|
| .0	      			| Turn left ahead					 			|
| .0				    | Priority road     							|

2. Right of way image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right-of-way at the next intersection 		| 
| .0    				| Slippery road									|
| .0					| Beware of ice/snow							|
| .0	      			| Road work					 					|
| .0				    | Bicycles crossing      						|

3. Yeild image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield   										| 
| .0    				| Keep right 									|
| .0					| Go straight or right							|
| .0	      			| Speed limit (60km/h)					 		|
| .0				    | Stop      									|

4. Keep right image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| keep right   									| 
| .0     				| Speed limit (50km/h)							|
| .0					| Slippery road									|
| .0	      			| Yeild					 						|
| .0				    | Speed limit (60km/h)      					|

5. Road work image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| Road work   									| 
| .6     				| Bicycles crossing								|
| .2					| Bumpy road									|
| .0	      			| Roundabout mandatory				 			|
| .0				    | Double curve      							|

6. Road work image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .75         			| Speed limit (70km/h)							| 
| .24     				| Speed limit (20km/h)							|
| .0					| Speed limit (30km/h)							|
| .0	      			| General caution					 			|
| .0				    | End of speed limit (80km/h)					|


References
[1] Traffic Sign Recognition with Multi-Scale Convolutional Networks. Pierre Sermanet and Yann LeCun, www.ieeexplore.ieee.org/iel5/6022827/6033131/06033589.pdf
[2] Feature scaling, https://en.wikipedia.org/wiki/Feature_scaling
[3] Transform function, https://nbviewer.jupyter.org/github/vxy10/