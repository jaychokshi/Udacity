#**Finding Lane Lines on the Road** 

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified
the draw_lines() function.

My pipeline consisted of 5 steps as below,

1. Convert RGB to grayscale
2. Apply a slight Gaussian blur
3. Perform Canny edge detection
4. Define a region of interest and mask away the undesired portions of the image
5. Retrieve Hough lines 
6. Apply lines to the original image


In order to draw a single line on the left and right lanes, I modified the draw_lines()
as below,

1. Figure out the set of left and right side of lanes using slope detection
2. Keep the mean of slopes and y-intercepts
3. based on two sets of slope and y-intercepts, figure out the x-intersection point
4. For each slope and intercept (+ve and -ve), using y=mx + b, figure out the
y-coordinates for top and bottom points of lines
5. Use CV to draw the lines between these coordinates

###2. Identify potential shortcomings with your current pipeline

1. The linear regression that is being used is tad coarse in definition, that is,
in the case of curves, I think a polynomial trendline may work better. Although, in that case,
we would have to apply multi variable regression technique

2. The size of region mask could be improved if we know the angle of plane the camera is
mounted with and the resolution, in order to make the selection more generic and apt.

3. For the challenge video, I explored a bunch of techniques to apply the HSV space and masks,
but wasn't able to cook up the working code before this submission timeline. May be, I could make
one more submission


###3. Suggest possible improvements to your pipeline

As mentioned in ###2 above, a better version of line regression and image massaging may improve
the quality of the pipeline
