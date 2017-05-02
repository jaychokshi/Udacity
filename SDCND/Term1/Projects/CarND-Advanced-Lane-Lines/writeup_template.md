## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./output_images/undistorted1.png "Undistorted checkerboard"
[image2]: ./output_images/undistorted2.png "Road Transformed"
[image3]: ./output_images/binary.png "Binary Example"
[image5]: ./output_images/sliding.jpg "sliding window"
[image7]: ./output_images/fit_poly1.png "Fit orig"
[image9]: ./output_images/fit_poly2.png "Fit warped"
[image6]: ./output_images/final.png "Final Output"
[image8]: ./output_images/curvature.png "Curvature"
[video1]: ./project_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook project.ipynb (# through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #217 through #229 in `extract_lane_edges()` api in `porject.py`).
I tried various combination of color spaces and gradient thresholds, using trial and error method. Finally, it looked like using red channel with horizontal gradient combination gives the acceptable results. I tried using magnitude and directional thresholds but that didn't help much. Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_perspective_transform()` on lines #205 to #210 on `project.py`. Also, the point extraction happens in `get_perspective_points()`, which appears in lines 190 through 202 in the file `project.py`.

```python
src = np.float32([[(w / 2) - top, h * h_scale],
                    [(w / 2) + top, h * h_scale],
                    [(w / 2) + bottom, h - h_sub],
                    [(w / 2) - bottom, h - h_sub]])

dst = np.float32([[(w / 2) - midpnt, (h / 2) - midpnt],
                    [(w / 2) + midpnt, (h / 2) - midpnt],
                    [(w / 2) + midpnt, (h / 2) + midpnt],
                    [(w / 2) - midpnt, (h / 2) + midpnt]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 570, 468      | 280, 0        |
| 710, 468      | 1000, 0       |
| 1010, 670     | 960, 720      |
| 270, 670      | 280, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

source points on base distorted image
![alt text][image7]

dest points on warped image
![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I used the code examples provided in the lectures to find identify the lane pixel and used CV2 fillpoly to draw the polynomial onto the image. I maintain the best fit for both left and right lanes during sliding window search. `sliding_window_fit_poly()` and `non_sliding_window_fit_poly()` apis (lines #236 to #376 in `project.py`) provide the code, that is based on lecutre notes.
Below image is an example:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Using the detected pixel points, I used the polynomial function to determine 2nd order fit. I used the US highway lane curvature parameters to covert from pixels to meters as advised in the lecture.
I implemented this step in lines #383 through #400 in my code in `project.py` in the function `lane_curvature()`.  Here is an example of my result on a test image:

![alt text][image8]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #407 through #428 in my code in `project.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
At this time, the pipeline doesn't fare well with varying degree of environment brightness, shadow, and haze conditions.
I think contrast detection, haze removal, and brightness normalization techniques should provide more robustness to the lane detection. People on the forum also have suggested to implement sort of confidence interval technique for both left and right fit to improve handling the tight turns (in last hard hard challange video).
