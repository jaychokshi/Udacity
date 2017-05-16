##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[img_car]: ./output_images/car.png
[img_not_car]: ./output_images/not_car.png
[img_car_hog]: ./output_images/car_hog.png
[img_not_car_hog]: ./output_images/not_car_hog.png
[img_feat_norm]: ./output_images/feature_normalized.png
[img_sld_win]: ./output_images/sliding_window.png
[img_sld_wins]: ./output_images/sliding_windows.png
[img_box_heat]: ./output_images/bboxes_and_heat.png
[img_lables]: ./output_images/labels_map.png
[img_op_boxes]: ./output_images/output_bboxes.png
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features()` from `line36` to `line53`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][img_car]
![alt text][img_not_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][img_car_hog]

To understand how an image without a car would look like in HOG, here is the HOG image as an exmaple for not-a-car image,

![alt text][img_not_car_hog]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of color spaces with HOG interpolation for feature extraction. I could obseve that YCrCb was pretty accurate (with HSV coming close enough). Initially I did use only Y channel, but found that it was not sufficient for detection quality, so I then used all three channels.

| Parameter               | Value    |
|:-----------------------:|:--------:|
| color_space             | YCrCb    |
| orientation             | 9        |
| pixels_per_cell         | 8        |
| cells_per_block         | 2        |
| hog channels            | 3        |
| spatial size            | (16, 16) |
| Histogram bins          | 16       |
| spatial feature used?   | Yes      |
| Histogram feature used? | Yes      |
| HOG feature used?       | Yes      |


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using three feature vectors,
1. spatial binning: To extract raw pixel color information
2. Color histogram: To add color spectrum information and,
3. HOG feature: to get the gradient info of the picture that helps extract the shapes of the object

The features are also scaled and normalized using StandardScaler method, sample result shown below,

![alt text][img_feat_norm]

The total number of feature are 6108. It takes 22.38 seconds to train classifier. The test accuracy of the model is 99.07%

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][img_sld_wins]

The sliding window algorithm provided in the lecture is used. It allows to search a car in a desired region of the frame with a desired window size. Each sub sampled window is scaled to 64x64 px before SVC classification.
The image/frame is also constricted to the interesting area where car could be found (i.e. remove sky, trees, etc).

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As mentioned before, I searched using YCrCb 3-channels with HOG features, and both spatially binning and color histograms. I also used false positives detection mechanism using heat map (showed in next section), which provided a nice result.  Here are an example image before heatmap detection,

![alt text][img_sld_win]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is an example frame and its corresponding heatmap:

![alt text][img_box_heat]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][img_lables]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][img_op_boxes]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project appears to be an interesting application of SVM, though with some shortcomings. When I tried to make my classifier quick (w/o using feature vector concatenation), it generated a lot of false positive detections (i.e. it triggers on other parts of an image that doesn't look like cars). It seems like it cannot generalize well, which kind of brings back to known issue with this kind of classifiers that the quality of the data fed while training makes huge difference. It would be really interesting to compare and contrast with CNNs to see if how they differ. There are some discussion on the forum about this, and I may actually take it upon to test it out.
