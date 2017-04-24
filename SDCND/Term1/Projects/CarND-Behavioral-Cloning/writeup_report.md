#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/center.jpg "Center driving"
[image21]: ./examples/recovery_center.jpg "Recovery Center Image"
[image22]: ./examples/recovery_left.jpg "Recovery Left Image"
[image23]: ./examples/recovery_right.jpg "Recovery Right Image"
[image3]: ./examples/normal.png "Normal Image"
[image4]: ./examples/flipped.png "Flipped Image"
[image5]: ./examples/processed.png "Cropped-Gaussia-Resized Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* run.mp4 video file captured using video.py
* https://youtu.be/T_qVYxcAwOU video file captured using iphone 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing.
```sh
python drive.py model.h5
```
Below is the video embedded link of the car driving on the Track:1

[![Video](https://img.youtube.com/vi/T_qVYxcAwOU/0.jpg)](https://youtu.be/T_qVYxcAwOU)


####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based off NVidia CNN model, that consists of a series of CNNs followed dense layers. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually
(model.py line 177).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a
combination of center lane driving, recovering from the left and right sides of
the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to is based on Nvidia's model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving
around track one. There were a few spots (curves, dust spots, etc) where the vehicle fell off the track.
To improve the driving behavior in these cases, I captured a more training examples on various curves on the track1 as a recovery data. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. 

####2. Final Model Architecture

The final model architecture (model.py line 135, def get_model_nvidia()) consisted of a convolution neural network with the following layers and layer sizes, below is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 66, 200, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
elu_1 (ELU)                  (None, 31, 98, 24)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
elu_2 (ELU)                  (None, 14, 47, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
elu_3 (ELU)                  (None, 5, 22, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
elu_4 (ELU)                  (None, 3, 20, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
elu_5 (ELU)                  (None, 1, 18, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300    
_________________________________________________________________
elu_6 (ELU)                  (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
elu_7 (ELU)                  (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
elu_8 (ELU)                  (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used Udacity provided data (that I realized is already a reversed driving), then I recorded a bunch of recovery data as advised in the lectures.
Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to course correct the path. Here is an example image of recovery.

![alt text][image21]
![alt text][image22]
![alt text][image23]

To augment the data set, I also flipped images, used both right and left images (with angles adjusted). 

![alt text][image3]
![alt text][image4]

I then preprocessed this data  by using crop method to remove unnecessary information from top and bottom of the image. I also applied "Gaussian blur" something that I learned from "traffic image" project, and resized for NVidia model.

![alt text][image5]

I finally randomly shuffled the data set and put 0.2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
